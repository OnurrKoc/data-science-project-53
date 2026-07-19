"""
DS-53 — Kaynaklı Asistan / RAG Değerlendirme

Senaryo: 27 gerçek Türkçe Wikipedia makalesinden oluşan bir doküman korpusu
üzerinde uçtan uca bir RAG (Retrieval-Augmented Generation) hattı kuruyorsun:
yükle -> parçala (chunk) -> indeksle (TF-IDF) -> getir (cosine similarity) ->
cevap üret. Bu proje sadece hattı KURMAKLA kalmıyor — retrieval kalitesini
gold set üzerinde ÖLÇÜYOR ve farklı chunk boyutlarını hit-rate açısından
KARŞILAŞTIRIYORSUN (ablation).

⚠️ Retrieval GERÇEK: scikit-learn'ün TfidfVectorizer'ı + cosine similarity ile
soruya en yakın parçaları bulursun (mock embed DEĞİL). Sadece `generate_answer`
fonksiyonundaki cevap üretimi mock LLM ile yapılır — `llm` dışarıdan parametre
olarak gelir (dependency injection), böylece testler API key gerektirmeden
deterministik çalışır.

Her fonksiyonun NotImplementedError satırını sil, kendi kodunu yaz. Testleri
çalıştır, hepsi geçene kadar iterate et: `python watch.py` veya
`pytest tests/test_question.py -v`
"""


# 1. Korpusu yükle
def load_corpus(path="data/corpus"):
    """
    `path` altındaki tüm .txt dosyalarını oku.

    Args:
        path: korpus klasörü (default "data/corpus")

    Returns:
        dict: {dosya_adi: metin} — 27 doküman
    """
    raise NotImplementedError("load_corpus henüz yazılmadı")


# 2. Tek bir metni parçala
def chunk_text(metin, chunk_size=100, overlap=20):
    """
    Metni KELİME bazlı, üst üste binen (overlap'li) parçalara böl.
    Her adımda `chunk_size` kelime al, sonra `chunk_size - overlap` kadar
    ilerle (komşu parçalar `overlap` kelimeyi paylaşır).

    Args:
        metin: str
        chunk_size: int — parça uzunluğu (kelime sayısı)
        overlap: int — komşu parçaların paylaştığı kelime sayısı

    Returns:
        list[str] — metin boşsa boş liste; kısa metin (chunk_size'dan az
        kelime) tek parça olarak döner.
    """
    raise NotImplementedError("chunk_text henüz yazılmadı")


# 3. Tüm korpusu parçala
def build_chunks(corpus, chunk_size=100, overlap=20):
    """
    Korpustaki her dokümanı `chunk_text` ile böl; her parçaya kaynağını
    (dosya adı) ekle.

    Args:
        corpus: dict — load_corpus çıktısı
        chunk_size, overlap: chunk_text'e geçilir

    Returns:
        list[dict] — her eleman {"text": str, "source": dosya_adi}
    """
    raise NotImplementedError("build_chunks henüz yazılmadı")


# 4. TF-IDF indeksi kur
def build_index(chunks):
    """
    Tüm chunk metinlerini TfidfVectorizer ile vektörleştir.

    Args:
        chunks: list[dict] — build_chunks çıktısı

    Returns:
        dict: {
            "vectorizer": fit edilmiş TfidfVectorizer,
            "matrix": fit_transform çıktısı (sparse matrix),
            "chunks": chunks (referans olarak sakla),
        }

    İpucu: sklearn.feature_extraction.text.TfidfVectorizer
    """
    raise NotImplementedError("build_index henüz yazılmadı")


# 5. Retrieval — en benzer k parçayı getir
def retrieve(soru, index, k=3):
    """
    Soruyu aynı vectorizer ile transform et, indeksteki tüm parçalarla
    COSINE SİMİLARİTY hesapla, en yüksek benzerlikli ilk `k` parçayı döndür.

    Args:
        soru: str
        index: dict — build_index çıktısı
        k: int — kaç parça getirileceği

    Returns:
        list[dict] — skor AZALAN sırada, her eleman
        {"text": str, "source": str, "score": float}

    İpucu: sklearn.metrics.pairwise.cosine_similarity
    """
    raise NotImplementedError("retrieve henüz yazılmadı")


# 6. Cevap üret (mock LLM ile)
def generate_answer(soru, retrieved, llm):
    """
    Getirilen parçalardan bir PROMPT oluştur (soru + bağlam), `llm(prompt)`
    çağırarak cevabı üret. `llm` dışarıdan gelir (dependency injection) —
    gerçek kullanımda bir LLM API'si, testlerde deterministik mock fonksiyon.

    Args:
        soru: str
        retrieved: list[dict] — retrieve çıktısı
        llm: callable — str (prompt) -> str (cevap)

    Returns:
        dict: {"cevap": str, "kaynaklar": list[str]} — kaynaklar, getirilen
        parçaların "source" alanlarının listesi (citation).

    İpucu: prompt'ta "BAĞLAM:" gibi bir ayraç kullan ki testler parse edebilsin.
    """
    raise NotImplementedError("generate_answer henüz yazılmadı")


# 7. Retrieval kalitesini ölç — hit-rate
def evaluate_hit_rate(gold_set, index, k=3):
    """
    `gold_set`'teki her örnek için `retrieve` çalıştır; `dogru_dosya`nın
    getirilen ilk k kaynak arasında olup olmadığına bak (HIT).

    Args:
        gold_set: list[dict] — her eleman
            {"soru": str, "dogru_dosya": str, "anahtar_kelime": str}
        index: dict — build_index çıktısı
        k: int

    Returns:
        dict: {"hit_rate": float (0-1), "hits": int, "total": int}
    """
    raise NotImplementedError("evaluate_hit_rate henüz yazılmadı")


# 8. Ablation — farklı chunk boyutlarını karşılaştır
def compare_chunk_sizes(corpus, gold_set, sizes=[30, 80, 300], k=1):
    """
    Her chunk boyutu için: parçala -> indeksle -> hit_rate hesapla.
    Bu, bitirme projesinin modül projelerinden ayrıştığı temel nokta —
    "kur"makla kalmayıp farklı stratejileri ÖLÇÜP KARŞILAŞTIRIYORSUN.

    NOT: Ablation'da bilerek `k=1` kullanıyoruz. Top-1 retrieval (yalnızca EN
    benzer parça) chunk boyutuna en duyarlı sinyali verir — küçük mü yoksa
    büyük chunk mı doğru dokümanı ilk sıraya koyuyor, en net burada görünür.
    (Retrieval'ın kendisi ve pipeline k=3 ile çalışır; bu sadece ölçüm ayarı.)

    Args:
        corpus: dict — load_corpus çıktısı
        gold_set: list[dict]
        sizes: list[int] — denenecek chunk boyutları (küçük/orta/büyük)
        k: int — ablation için top-k (default 1, en ayırt edici)

    Returns:
        dict: {boyut: hit_rate} — ör. {30: 0.93, 80: 0.85, 300: 0.74}

    İpucu: overlap'i sabit (örn. 20) tutabilirsin, sadece chunk_size'ı değiştir.
    Sonuçları yorumla: bu korpusta konu kümeleri (astronomi: Yıldız/Galaksi/
    Gezegen/Ay... ; biyoloji: Hücre/Virüs/Enzim/DNA...) kelime paylaştığı için
    retrieval karışabilir; hangi chunk boyutunun neden kazandığını tartış.
    """
    raise NotImplementedError("compare_chunk_sizes henüz yazılmadı")


# 9. Uçtan uca pipeline
def run_pipeline(soru, corpus_path="data/corpus", chunk_size=100, overlap=20, k=3):
    """
    Tüm hattı uçtan uca çalıştır: load_corpus -> build_chunks -> build_index
    -> retrieve -> generate_answer (mock LLM ile).

    Args:
        soru: str
        corpus_path, chunk_size, overlap, k: hattın parametreleri

    Returns:
        dict: {
            "soru": str,
            "retrieved": list[dict] (retrieve çıktısı),
            "cevap": str,
            "kaynaklar": list[str],
        }

    İpucu: generate_answer için basit bir mock llm tanımlayabilirsin, örn:
        def mock_llm(prompt):
            return "Verilen kaynaklara göre: " + prompt.split("BAĞLAM:")[-1][:120]
    """
    raise NotImplementedError("run_pipeline henüz yazılmadı")


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"📚 Korpus yüklendi: {len(corpus)} doküman")

    chunks = build_chunks(corpus)
    print(f"✂️  Toplam chunk sayısı: {len(chunks)}")

    index = build_index(chunks)

    import json
    with open("data/gold_set.json", encoding="utf-8") as f:
        gold_set = json.load(f)

    metrik = evaluate_hit_rate(gold_set, index, k=3)
    print(f"📊 Hit-rate (chunk_size=100, k=3): {metrik}")

    karsilastirma = compare_chunk_sizes(corpus, gold_set)
    print(f"📊 Chunk boyutu karşılaştırması (ablation, k=1): {karsilastirma}")

    sonuc = run_pipeline("Bitkilerin ışık enerjisini kimyasal enerjiye çevirdiği işlem nedir?")
    print(f"🤖 Örnek cevap: {sonuc['cevap']}")
    print(f"🔗 Kaynaklar: {sonuc['kaynaklar']}")
