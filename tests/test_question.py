import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.task_manager import (
    load_corpus,
    chunk_text,
    build_chunks,
    build_index,
    retrieve,
    generate_answer,
    evaluate_hit_rate,
    compare_chunk_sizes,
    run_pipeline,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus")
GOLD_PATH = os.path.join(BASE_DIR, "data", "gold_set.json")


# ──────────────────────────────────────────────────────
# MOCK LLM — deterministik, key gerektirmez.
# Retrieval (TF-IDF + cosine) GERÇEK; sadece cevap üretimi mock'lanır.
# ──────────────────────────────────────────────────────
def mock_llm(prompt):
    return "Verilen kaynaklara göre: " + prompt.split("BAĞLAM:")[-1][:120]


@pytest.fixture(scope="module")
def gold_set():
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(CORPUS_PATH)


@pytest.fixture(scope="module")
def chunks(corpus):
    return build_chunks(corpus, chunk_size=100, overlap=20)


@pytest.fixture(scope="module")
def index(chunks):
    return build_index(chunks)


# 1. load_corpus — 27 doküman
def test_load_corpus(corpus):
    assert isinstance(corpus, dict)
    assert len(corpus) == 27
    assert "Fotosentez.txt" in corpus
    assert all(isinstance(metin, str) and len(metin) > 0 for metin in corpus.values())


# 2. chunk_text — overlap'li, boş değil, boyut ~chunk_size
def test_chunk_text_temel():
    metin = " ".join(f"kelime{i}" for i in range(250))
    parcalar = chunk_text(metin, chunk_size=100, overlap=20)
    assert len(parcalar) >= 2
    for p in parcalar:
        assert len(p.split()) <= 100
    assert all(len(p) > 0 for p in parcalar)


def test_chunk_text_overlap():
    metin = " ".join(f"kelime{i}" for i in range(250))
    parcalar = chunk_text(metin, chunk_size=100, overlap=20)
    # ilk parçanın son kelimeleri, ikinci parçanın ilk kelimeleriyle örtüşmeli
    ilk_son_kelimeler = parcalar[0].split()[-20:]
    ikinci_ilk_kelimeler = parcalar[1].split()[:20]
    assert ilk_son_kelimeler == ikinci_ilk_kelimeler


def test_chunk_text_kisa_metin():
    kisa = "kısa bir metin"
    parcalar = chunk_text(kisa, chunk_size=100, overlap=20)
    assert parcalar == [kisa]


def test_chunk_text_bos_metin():
    assert chunk_text("", chunk_size=100, overlap=20) == []


# 3. build_chunks — her chunk'ta source var, source korpus dosyalarından
def test_build_chunks(corpus, chunks):
    assert len(chunks) >= len(corpus)
    assert all(set(c.keys()) == {"text", "source"} for c in chunks)
    kaynaklar = {c["source"] for c in chunks}
    assert kaynaklar == set(corpus.keys())


# 4. build_index — matrix satır sayısı = chunk sayısı
def test_build_index(chunks, index):
    assert index["matrix"].shape[0] == len(chunks)
    assert len(index["chunks"]) == len(chunks)


# 5. retrieve — k sonuç, azalan sırada skor, geçerli source
def test_retrieve_k_ve_sira(index, corpus):
    sonuc = retrieve("fotosentez sırasında klorofil ışığı nasıl emer?", index, k=3)
    assert len(sonuc) == 3
    skorlar = [r["score"] for r in sonuc]
    assert skorlar == sorted(skorlar, reverse=True)
    assert all(r["source"] in corpus for r in sonuc)
    assert all(set(r.keys()) == {"text", "source", "score"} for r in sonuc)


def test_retrieve_en_iyi_kaynak(index):
    sonuc = retrieve("Fotosentezde ışık enerjisini emen yeşil pigmentin adı nedir?", index, k=1)
    assert sonuc[0]["source"] == "Fotosentez.txt"


def test_retrieve_farkli_k(index):
    assert len(retrieve("bal arısı nedir?", index, k=5)) == 5
    assert len(retrieve("bal arısı nedir?", index, k=1)) == 1


# 6. generate_answer — mock llm ile kaynaklar dolu, cevap string
def test_generate_answer(index):
    retrieved = retrieve("DNA nükleotitlerden mi oluşur?", index, k=3)
    sonuc = generate_answer("DNA nükleotitlerden mi oluşur?", retrieved, mock_llm)
    assert set(sonuc.keys()) == {"cevap", "kaynaklar"}
    assert isinstance(sonuc["cevap"], str) and len(sonuc["cevap"]) > 0
    assert len(sonuc["kaynaklar"]) == 3
    assert all(isinstance(k, str) for k in sonuc["kaynaklar"])


# 7. evaluate_hit_rate — gerçek gold set üzerinde >= 0.70
def test_evaluate_hit_rate(index, gold_set):
    metrik = evaluate_hit_rate(gold_set, index, k=3)
    assert set(metrik.keys()) == {"hit_rate", "hits", "total"}
    assert metrik["total"] == len(gold_set)
    assert 0.0 <= metrik["hit_rate"] <= 1.0
    assert metrik["hit_rate"] >= 0.70


# 8. compare_chunk_sizes — ablation, 3 boyut için skor
# k=1 ile chunk boyutu hit_rate'i belirgin etkiler (küçük chunk bilgiyi konsantre
# tutar, büyük chunk seyreltir) — öğrenci bu farkı raporunda analiz eder.
def test_compare_chunk_sizes(corpus, gold_set):
    sonuc = compare_chunk_sizes(corpus, gold_set, sizes=[30, 80, 300], k=1)
    assert set(sonuc.keys()) == {30, 80, 300}
    for skor in sonuc.values():
        assert 0.0 <= skor <= 1.0


# 9. Citation / faithfulness — getirilen doğru kaynağın chunk'ında anahtar kelime geçmeli
def test_citation_anahtar_kelime(index, gold_set):
    en_az_bir_dogru = False
    for ornek in gold_set:
        getirilen = retrieve(ornek["soru"], index, k=3)
        for r in getirilen:
            if r["source"] == ornek["dogru_dosya"] and ornek["anahtar_kelime"].lower() in r["text"].lower():
                en_az_bir_dogru = True
                break
        if en_az_bir_dogru:
            break
    assert en_az_bir_dogru


# 10. run_pipeline — uçtan uca
def test_run_pipeline():
    sonuc = run_pipeline("Fotosentezde ışık enerjisini emen yeşil pigmentin adı nedir?")
    assert set(sonuc.keys()) == {"soru", "retrieved", "cevap", "kaynaklar"}
    assert len(sonuc["retrieved"]) == 3
    assert isinstance(sonuc["cevap"], str) and len(sonuc["cevap"]) > 0
    assert len(sonuc["kaynaklar"]) == 3


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────
import requests


def _send_score(user_score):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return
    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return
    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {"user_id": USER_ID, "project_id": PROJECT_ID, "user_score": user_score, "is_auto": True}
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
