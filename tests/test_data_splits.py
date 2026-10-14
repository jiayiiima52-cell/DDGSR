import scipy.sparse as sp
import pytest

import ddgsr.data as data


def _pairs(split):
    return set(zip(split["users"], split["pos_item"]))


@pytest.mark.parametrize("dataset", ["tssgcf", "programmableweb"])
def test_validation_graphs_use_only_training_interactions(tmp_path, monkeypatch, dataset):
    if dataset == "tssgcf":
        folder = tmp_path / "tssgcf_official" / "Data"
    else:
        folder = tmp_path / "programmableweb-onlyInteraction"
    folder.mkdir(parents=True)
    (folder / "train.txt").write_text("0 0 1\n1 1 2\n", encoding="utf-8")

    negative_sources = []

    def capture_negative_source(split, users, items, seed):
        negative_sources.append(_pairs(split))
        return sp.csr_matrix((users + items, users + items), dtype="float32")

    monkeypatch.setattr(data, "_sampled_negative_graph", capture_negative_source)
    train, validation, test, _, _, _ = data.load_dataset(
        tmp_path, dataset, full_train=False, load_test=False
    )

    assert test is None
    assert _pairs(train).isdisjoint(_pairs(validation))
    assert negative_sources == [_pairs(train)]
    assert negative_sources[0].isdisjoint(_pairs(validation))


def test_tssgcf_full_train_graph_uses_all_training_interactions(tmp_path, monkeypatch):
    folder = tmp_path / "tssgcf_official" / "Data"
    folder.mkdir(parents=True)
    (folder / "train.txt").write_text("0 0 1\n1 1 2\n", encoding="utf-8")

    negative_sources = []

    def capture_negative_source(split, users, items, seed):
        negative_sources.append(_pairs(split))
        return sp.csr_matrix((users + items, users + items), dtype="float32")

    monkeypatch.setattr(data, "_sampled_negative_graph", capture_negative_source)
    train, validation, _, _, _, _ = data.load_dataset(
        tmp_path, "tssgcf", full_train=True, load_test=False
    )

    assert _pairs(train) == {(0, 0), (0, 1), (1, 1), (1, 2)}
    assert _pairs(validation) == _pairs(train)
    assert negative_sources == [_pairs(train)]
