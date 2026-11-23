import json
import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def metin_temizle(metin: str) -> str:
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


class BilgiTR_RAG:
    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.parcalar = []
        self.tfidf_matrisi = None

    def _parcala(self, metin: str, dosya_adi: str) -> list:
        konu = dosya_adi.replace(".txt", "").replace("_", " ")
        kelimeler = metin.split()
        parcalar = []
        adim = max(1, self.chunk_size - self.overlap)

        for i in range(0, len(kelimeler), adim):
            parca_kelimeler = kelimeler[i : i + self.chunk_size]
            if len(parca_kelimeler) < 15 and parcalar:
                break
            ham_metin = f"{konu}: " + " ".join(parca_kelimeler)
            parcalar.append({
                "kaynak": dosya_adi,
                "parca_id": len(parcalar),
                "ham_metin": ham_metin,
                "temiz_metin": metin_temizle(ham_metin)
            })
        return parcalar

    def indeksle(self, corpus_klasoru: str = "data/corpus"):
        self.parcalar = []
        dosyalar = sorted(Path(corpus_klasoru).glob("*.txt"))
        if not dosyalar:
            raise FileNotFoundError(f"'{corpus_klasoru}' klasöründe .txt dosyası bulunamadı.")

        for dosya in dosyalar:
            icerik = dosya.read_text(encoding="utf-8")
            self.parcalar.extend(self._parcala(icerik, dosya.name))

        temiz_metinler = [p["temiz_metin"] for p in self.parcalar]
        self.tfidf_matrisi = self.vectorizer.fit_transform(temiz_metinler)

    def getir(self, soru: str, top_k: int = 3) -> list:
        temiz_soru = metin_temizle(soru)
        soru_vektoru = self.vectorizer.transform([temiz_soru])
        skorlar = cosine_similarity(soru_vektoru, self.tfidf_matrisi).flatten()
        en_iyi_indeksler = np.argsort(skorlar)[::-1][:top_k]

        sonuclar = []
        for idx in en_iyi_indeksler:
            sonuclar.append({
                "kaynak": self.parcalar[idx]["kaynak"],
                "parca_id": self.parcalar[idx]["parca_id"],
                "skor": round(float(skorlar[idx]), 4),
                "metin": self.parcalar[idx]["ham_metin"]
            })
        return sonuclar

    def cevapla(self, soru: str, top_k: int = 3) -> dict:
        getirilenler = self.getir(soru, top_k=top_k)
        kaynaklar = list(dict.fromkeys([g["kaynak"] for g in getirilenler]))
        cevap_satirlari = [
            f"• {g['metin'][:200]}... [Kaynak: {g['kaynak']} | Skor: {g['skor']}]"
            for g in getirilenler
        ]
        return {
            "soru": soru,
            "cevap": "\n".join(cevap_satirlari),
            "kaynaklar": kaynaklar
        }


def degerlendir(rag: BilgiTR_RAG, gold_yolu: str = "data/gold_set.json", top_k: int = 3) -> dict:
    with open(gold_yolu, "r", encoding="utf-8") as f:
        gold_set = json.load(f)

    isabet = 0
    mrr_toplam = 0.0
    anahtar_isabet = 0

    for veri in gold_set:
        soru = veri["soru"]
        dogru_dosya = veri["dogru_dosya"]
        anahtar = metin_temizle(veri["anahtar_kelime"])

        getirilenler = rag.getir(soru, top_k=top_k)
        getirilen_dosyalar = [g["kaynak"] for g in getirilenler]
        birlesik_metin = metin_temizle(" ".join([g["metin"] for g in getirilenler]))

        if dogru_dosya in getirilen_dosyalar:
            isabet += 1
            sira = getirilen_dosyalar.index(dogru_dosya) + 1
            mrr_toplam += 1.0 / sira

        if anahtar in birlesik_metin:
            anahtar_isabet += 1

    n = len(gold_set)
    return {
        "hit_rate": round(isabet / n, 3),
        "mrr": round(mrr_toplam / n, 3),
        "keyword_hit_rate": round(anahtar_isabet / n, 3),
        "oran": f"{isabet}/{n}"
    }


if __name__ == "__main__":
    print("=== BilgiTR Kaynaklı Soru-Cevap Sistemi (RAG) ===")
    rag = BilgiTR_RAG(chunk_size=100, overlap=20)
    rag.indeksle("data/corpus")
    print(f"İndekslenen toplam parça sayısı: {len(rag.parcalar)}\n")

    print("--- Gold Set Değerlendirme Sonuçları ---")
    for k in (1, 3, 5):
        sonuc = degerlendir(rag, "data/gold_set.json", top_k=k)
        print(
            f"k={k} | Hit-Rate@{k}: {sonuc['hit_rate']:.3f} ({sonuc['oran']}) | "
            f"MRR: {sonuc['mrr']:.3f} | Anahtar Kelime İsabeti: {sonuc['keyword_hit_rate']:.3f}"
        )

    print("\n--- Örnek Soru ve Kaynaklı Cevap ---")
    ornek_soru = "Genetik bilgiyi taşıyan, çift sarmal yapıdaki nükleik asit molekülü nedir?"
    yanit = rag.cevapla(ornek_soru, top_k=3)
    print(f"Soru: {yanit['soru']}")
    print(f"Cevap:\n{yanit['cevap']}")
    print(f"Kullanılan Kaynaklar: {', '.join(yanit['kaynaklar'])}")
