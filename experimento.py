#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def find_member(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zf.namelist() if name.replace("\\", "/").endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"Não encontrei '{suffix}' dentro do ZIP KuaiRec.")
    # Prefer the expected data folder if several copies exist.
    matches.sort(key=lambda n: ("/data/" not in n.replace("\\", "/"), len(n)))
    return matches[0]


def parse_feature_list(value: object) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return []
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: List[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def row_minmax_candidate(scores: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Min-max scale each user's candidate scores; training items are ignored."""
    scaled = np.zeros(scores.shape, dtype=np.float32)
    for u in range(scores.shape[0]):
        candidates = ~train_mask[u]
        if not np.any(candidates):
            continue
        values = scores[u, candidates].astype(np.float32, copy=False)
        low = float(np.min(values))
        high = float(np.max(values))
        if high > low:
            scaled[u, candidates] = (values - low) / (high - low)
    return scaled


def dcg_at_k(binary_relevance: np.ndarray, k: int) -> float:
    rel = binary_relevance[:k].astype(float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def evaluate_model(
    name: str,
    scores: np.ndarray,
    train_mask: np.ndarray,
    test_groups: Dict[int, pd.DataFrame],
    item_to_idx: Dict[int, int],
    item_features: csr_matrix,
    k: int = 10,
) -> Tuple[dict, List[dict], set]:
    n_users, n_items = scores.shape
    per_user: List[dict] = []
    recommended_union: set = set()

    for u_idx, (raw_user, test_rows) in enumerate(test_groups.items()):
        user_scores = scores[u_idx].astype(np.float32, copy=True)
        user_scores[train_mask[u_idx]] = -np.inf
        available = np.isfinite(user_scores)
        if int(available.sum()) < k:
            continue

        top = np.argpartition(user_scores, -k)[-k:]
        top = top[np.argsort(user_scores[top])[::-1]]
        recommended_union.update(int(x) for x in top)

        relevant = set()
        for row in test_rows.itertuples(index=False):
            # watch_ratio >= 1 indicates play_duration >= video_duration.
            if float(row.watch_ratio) >= 1.0:
                idx = item_to_idx.get(int(row.video_id))
                if idx is not None:
                    relevant.add(idx)

        rel_vec = np.asarray([1 if int(item) in relevant else 0 for item in top], dtype=int)
        hits = int(rel_vec.sum())
        precision = hits / float(k)
        recall = hits / float(len(relevant)) if relevant else np.nan
        if relevant:
            dcg = dcg_at_k(rel_vec, k)
            ideal = np.zeros(min(k, len(relevant)), dtype=int)
            ideal[:] = 1
            idcg = dcg_at_k(ideal, k)
            ndcg = dcg / idcg if idcg > 0 else 0.0
        else:
            ndcg = np.nan

        # Intra-list diversity = average pairwise cosine distance over category vectors.
        top_features = item_features[top]
        sim = cosine_similarity(top_features, dense_output=True)
        upper = sim[np.triu_indices(k, k=1)]
        ild = float(np.mean(1.0 - upper)) if upper.size else 0.0

        per_user.append({
            "user_id": int(raw_user),
            "precision_at_k": precision,
            "recall_at_k": recall,
            "ndcg_at_k": ndcg,
            "ild_at_k": ild,
            "relevant_test_items": len(relevant),
            "hits_at_10": hits,
        })

    user_df = pd.DataFrame(per_user)
    if user_df.empty:
        raise RuntimeError(f"Nenhum usuário elegível foi avaliado para o modelo {name}.")

    summary = {
        "model": name,
        "precision_at_k": float(user_df["precision_at_k"].mean()),
        "recall_at_k": float(user_df["recall_at_k"].mean()),
        "ndcg_at_k": float(user_df["ndcg_at_k"].mean()),
        "ild_at_k": float(user_df["ild_at_k"].mean()),
        "catalog_coverage_at_k": float(len(recommended_union) / n_items),
        "users_evaluated": int(len(user_df)),
        "unique_items_recommended": int(len(recommended_union)),
    }
    return summary, per_user, recommended_union


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", required=True, help="Caminho para KuaiRec.zip")
    parser.add_argument("--output", default="resultados_kuairec", help="Pasta de saída")
    parser.add_argument("--k", type=int, default=10, help="Tamanho da lista recomendada (padrão: 10)")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Proporção temporal de treino (padrão: 0.8)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Peso do modelo colaborativo no híbrido (padrão: 0.5)")
    args = parser.parse_args()

    if not 0.5 <= args.train_fraction < 1.0:
        parser.error("--train-fraction deve estar entre 0.5 (inclusive) e 1.0 (exclusive).")
    if args.k < 2:
        parser.error("--k deve ser pelo menos 2 para calcular diversidade intra-lista.")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha deve estar entre 0 e 1.")

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        parser.error(f"Arquivo não encontrado: {zip_path}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Lendo os arquivos oficiais do KuaiRec...")
    with zipfile.ZipFile(zip_path) as zf:
        matrix_member = find_member(zf, "small_matrix.csv")
        categories_member = find_member(zf, "item_categories.csv")
        with zf.open(matrix_member) as f:
            interactions = pd.read_csv(
                f,
                usecols=["user_id", "video_id", "timestamp", "watch_ratio"],
            )
        with zf.open(categories_member) as f:
            categories = pd.read_csv(f, converters={"feat": parse_feature_list})

    interactions = interactions.dropna(subset=["user_id", "video_id", "timestamp", "watch_ratio"])
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["video_id"] = interactions["video_id"].astype(int)
    interactions["timestamp"] = pd.to_numeric(interactions["timestamp"], errors="coerce")
    interactions["watch_ratio"] = pd.to_numeric(interactions["watch_ratio"], errors="coerce")
    interactions = interactions.dropna(subset=["timestamp", "watch_ratio"])
    # Defensive aggregation in case a user-video pair appears more than once.
    interactions = (
        interactions.groupby(["user_id", "video_id"], as_index=False)
        .agg(timestamp=("timestamp", "max"), watch_ratio=("watch_ratio", "mean"))
    )

    user_ids = np.sort(interactions["user_id"].unique())
    item_ids = np.sort(interactions["video_id"].unique())
    user_to_idx = {int(v): i for i, v in enumerate(user_ids)}
    item_to_idx = {int(v): i for i, v in enumerate(item_ids)}
    interactions["u_idx"] = interactions["user_id"].map(user_to_idx).astype(int)
    interactions["i_idx"] = interactions["video_id"].map(item_to_idx).astype(int)

    print("[2/7] Separando treino e teste cronologicamente por usuário...")
    interactions = interactions.sort_values(["user_id", "timestamp", "video_id"])
    train_parts = []
    test_groups: Dict[int, pd.DataFrame] = {}
    for raw_user, group in interactions.groupby("user_id", sort=True):
        if len(group) < 5:
            continue
        cut = int(np.floor(len(group) * args.train_fraction))
        cut = min(max(cut, 1), len(group) - 1)
        train_parts.append(group.iloc[:cut])
        test_groups[int(raw_user)] = group.iloc[cut:][["video_id", "watch_ratio"]].copy()
    if not train_parts:
        raise RuntimeError("Não foi possível montar a divisão treino/teste.")
    train = pd.concat(train_parts, ignore_index=True)
    evaluated_users = list(test_groups.keys())
    # Reindex users to the evaluated subset only.
    active_user_ids = np.asarray(evaluated_users, dtype=int)
    active_user_to_idx = {int(v): i for i, v in enumerate(active_user_ids)}
    train = train[train["user_id"].isin(active_user_to_idx)].copy()
    train["u_active"] = train["user_id"].map(active_user_to_idx).astype(int)

    n_users = len(active_user_ids)
    n_items = len(item_ids)
    # log1p moderates extreme viewing ratios without turning feedback into explicit likes.
    values = np.log1p(np.clip(train["watch_ratio"].to_numpy(dtype=np.float32), 0.0, None))
    R = csr_matrix((values, (train["u_active"].to_numpy(), train["i_idx"].to_numpy())), shape=(n_users, n_items), dtype=np.float32)
    train_mask = np.zeros((n_users, n_items), dtype=bool)
    train_mask[train["u_active"].to_numpy(), train["i_idx"].to_numpy()] = True

    print("[3/7] Construindo os vetores de categorias dos vídeos...")
    category_map = {}
    category_lists = {}
    for row in categories.itertuples(index=False):
        try:
            vid = int(row.video_id)
        except (TypeError, ValueError):
            continue
        feats = parse_feature_list(getattr(row, "feat", []))
        category_lists[vid] = feats
    all_features = sorted({f for feats in category_lists.values() for f in feats})
    feat_to_idx = {int(v): i for i, v in enumerate(all_features)}
    rows, cols, data = [], [], []
    for vid, feats in category_lists.items():
        i_idx = item_to_idx.get(int(vid))
        if i_idx is None:
            continue
        for feat in feats:
            j = feat_to_idx.get(int(feat))
            if j is not None:
                rows.append(i_idx)
                cols.append(j)
                data.append(1.0)
    if not data:
        raise RuntimeError("Não foi possível construir os atributos de categoria dos vídeos.")
    X = csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(n_items, len(feat_to_idx)), dtype=np.float32)
    X_norm = normalize(X, norm="l2", axis=1)

    print("[4/7] Calculando as pontuações colaborativas...")
    item_sim = cosine_similarity(R.T, dense_output=True).astype(np.float32)
    np.fill_diagonal(item_sim, 0.0)
    scores_cf = np.asarray(R @ item_sim, dtype=np.float32)
    del item_sim

    print("[5/7] Calculando as pontuações baseadas em conteúdo...")
    user_profiles = R @ X
    user_profiles = normalize(user_profiles, norm="l2", axis=1)
    scores_cb = (user_profiles @ X_norm.T).toarray().astype(np.float32)

    print("[6/7] Formando o modelo híbrido e avaliando os três modelos...")
    cf_norm = row_minmax_candidate(scores_cf, train_mask)
    cb_norm = row_minmax_candidate(scores_cb, train_mask)
    scores_hybrid = args.alpha * cf_norm + (1.0 - args.alpha) * cb_norm

    models = [
        ("Filtragem colaborativa (item-item)", scores_cf),
        ("Filtragem baseada em conteúdo", scores_cb),
        (f"Híbrido (alpha={args.alpha:.2f})", scores_hybrid),
    ]
    summaries = []
    user_results = []
    for name, model_scores in models:
        summary, per_user, _ = evaluate_model(
            name, model_scores, train_mask, test_groups, item_to_idx, X, k=args.k
        )
        summaries.append(summary)
        for row in per_user:
            row["model"] = name
            user_results.append(row)
        print(f"  - {name}: NDCG@{args.k}={summary['ndcg_at_k']:.4f}; ILD@{args.k}={summary['ild_at_k']:.4f}")

    print("[7/7] Salvando os resultados...")
    summary_df = pd.DataFrame(summaries)
    users_df = pd.DataFrame(user_results)
    summary_path = output_dir / "comparacao_modelos.csv"
    users_path = output_dir / "metricas_por_usuario.csv"
    metadata_path = output_dir / "metadados_execucao.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    users_df.to_csv(users_path, index=False, encoding="utf-8-sig")
    metadata = {
        "dataset": "KuaiRec small_matrix.csv + item_categories.csv",
        "users_in_matrix": int(len(user_ids)),
        "items_in_matrix": int(len(item_ids)),
        "interactions_after_pair_aggregation": int(len(interactions)),
        "users_evaluated": int(n_users),
        "training_fraction_per_user_chronological": float(args.train_fraction),
        "k": int(args.k),
        "hybrid_collaborative_weight_alpha": float(args.alpha),
        "relevance_proxy": "watch_ratio >= 1.0 (complete view; not an explicit like)",
        "feedback_transform": "log1p(max(watch_ratio, 0))",
        "models": [x[0] for x in models],
        "metrics": ["Precision@K", "Recall@K", "NDCG@K", "ILD@K", "CatalogCoverage@K"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResumo salvo em:", summary_path.resolve())
    print(summary_df.to_string(index=False))
    print("\nArquivos adicionais:", users_path.resolve(), "e", metadata_path.resolve())
    print("\nIMPORTANTE: interpretar watch_ratio >= 1.0 como proxy de visualização completa, não como like/satisfação.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Execução interrompida pelo usuário.", file=sys.stderr)
        raise SystemExit(130)
