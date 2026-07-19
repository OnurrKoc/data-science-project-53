# DS-53 — Kaynaklı Asistan / RAG Değerlendirme (Bitirme Projesi)

**Modül**: Bitirme Projesi (RAG + Değerlendirme) • **Süre**: 4-6 saat

## 🎯 Proje Senaryosu

27 gerçek Türkçe Wikipedia makalesinden (Fotosentez, DNA, Kara delik, Osmanlı
İmparatorluğu, Gezegen, Yıldız, İnternet, Antibiyotik, Kahve, Everest Dağı...
toplam ~10000 kelime) oluşan bir doküman korpusu üzerinde uçtan uca bir
**Kaynaklı Asistan (RAG)** kuruyorsun: kullanıcı doğal dille bir soru soruyor,
sistem korpustan ilgili parçaları getirip kaynak göstererek cevap üretiyor.

Daha önceki RAG projelerinde (DS-49, DS-50, DS-51) küçük kurgusal metinler ve
mock (sahte) embedding kullandın. **Bu projede fark şu:**

1. **Gerçek, büyük korpus** — 27 gerçek Wikipedia makalesi, kurgusal değil.
   Bazıları konu olarak yakın (astronomi: Güneş Sistemi/Gezegen/Yıldız/Galaksi;
   biyoloji: DNA/Hücre/Virüs/Enzim) — retrieval'ı zorlar, ölçüm anlamlı olur.
2. **Gerçek retrieval** — `TfidfVectorizer` (scikit-learn) + cosine similarity
   ile GERÇEK bir vektör uzayında arama yapıyorsun. Mock embed YOK.
3. **Ölçüm + karşılaştırma odaklı** — Sadece "kur"muyorsun; retrieval
   kalitesini bir **gold set** üzerinde ölçüyor (hit-rate), farklı **chunk
   boyutlarını karşılaştırıyorsun (ablation)**. Bu, modül projelerinde YOK —
   bitirme projesinin ayırt edici noktası.
4. **Mock-LLM sadece cevap üretiminde** — retrieval matematiği tamamen
   gerçek; sadece son adımdaki LLM çağrısı (metin üretimi) test edilebilir
   olması için mock'lanır (dependency injection deseni).

## 🧩 Kuracağın Hat

1. `load_corpus` — `data/corpus/` altındaki 27 .txt dosyasını yükle
2. `chunk_text` / `build_chunks` — kelime bazlı, overlap'li parçalama; her
   parçaya kaynağını (dosya adı) ekle
3. `build_index` — tüm parçaları `TfidfVectorizer` ile vektörleştir
4. `retrieve` — soruyu vektörleştir, cosine similarity ile en yakın k parçayı
   getir
5. `generate_answer` — getirilen parçalardan bir prompt kur, mock LLM ile
   cevap üret, kaynakları (citation) döndür
6. `evaluate_hit_rate` — bir **gold set** (`data/gold_set.json`) üzerinde
   retrieval'ın doğru dokümanı getirip getirmediğini ölç
7. `compare_chunk_sizes` — **ablation**: 30/80/300 kelimelik chunk
   boyutlarını dene, her biri için hit-rate hesapla, karşılaştır
8. `run_pipeline` — tüm hattı uçtan uca çalıştır

## 📏 Neden Ölçüyoruz? (önemli)

Bir RAG sisteminin "iyi çalıştığını" söylemek yetmez — **sayı vermen**
gerekir. `gold_set.json`, gerçek korpus içeriğine göre yazılmış 27 soru-cevap
çifti içerir: her biri bir soru, doğru kaynağı (`dogru_dosya`) ve cevapta
geçmesi beklenen bir anahtar kelime (`anahtar_kelime`, citation/faithfulness
kontrolü için).

**Hit-rate**: her soru için `retrieve` çalıştırıp, doğru dosyanın getirilen
ilk k sonuç içinde olup olmadığına bakarsın. Oran, sistemin ne kadar doğru
kaynağı bulduğunu gösterir (0-1 arası).

**Chunk boyutu karşılaştırması (ablation)**: chunk ne kadar büyük/küçük
olmalı? Küçük chunk'lar daha odaklı ama bağlamdan yoksun olabilir; büyük
chunk'lar daha fazla bağlam taşır ama alakasız bilgiyle "sulanabilir". Bunu
tahmin etmek yerine 3 farklı boyutu (30, 80, 300 kelime) deneyip hit-rate'i
ÖLÇEREK karşılaştırırsın — bu, bitirme projesinde sana "kur"maktan bir adım
öteye, gerçek bir RAG mühendisinin yaptığı **değerlendir + karşılaştır +
raporla** işine geçiriyor. (İpucu: k=1 ile küçük chunk'ların büyüklere göre
neden daha iyi sonuç verebildiğini raporunda tartış.)

## 🔌 llm Nasıl Çalışıyor? (önemli)

`generate_answer` fonksiyonunun `llm`'i **parametre olarak alır**
(dependency injection):

- **`llm`**: prompt (str) → cevap (str). Gerçekte bir LLM sağlayıcısının
  API çağrısı olur.
- **Testlerde** deterministik sahte bir `llm` fonksiyonu verilir → API key
  GEREKMEZ, cevap her zaman aynı.
- Retrieval (adım 3-5) tamamen GERÇEK — sadece son adımdaki metin üretimi
  mock'lanır.

Aynı kod, sadece verdiğin fonksiyon değişir — test edilebilir RAG kodu
yazmanın profesyonel yolu budur.

## 📦 Kurulum

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python watch.py                 # otomatik test (dosya değişince)
# veya
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

`USER_ID`'ni Kaizu profilinden alıp yaz (`PROJECT_ID`'ye dokunma). Testler
geçince skor otomatik gider.

## 🔑 API key? (opsiyonel)

Puanlama için **gerekmez** (mock LLM). Gerçek bir LLM ile denemek istersen
`cp .env.example .env` + kendi anahtarını koy (`.env` asla commit edilmez).

## 💡 İpuçları

- `chunk_text`'te sonsuz döngüye düşmemek için ilerleme adımını
  `max(1, chunk_size - overlap)` şeklinde hesapla.
- `retrieve`'de `cosine_similarity` çıktısı bir matris döner (1×N) — ilk
  satırı al, sonra `argsort()[::-1][:k]` ile en yüksek k indeksi bul.
- `evaluate_hit_rate` ve `compare_chunk_sizes` birbirini kullanabilir —
  `compare_chunk_sizes` her boyut için yeniden `build_chunks` + `build_index`
  + `evaluate_hit_rate` çağırır.
- Sonuçları yorumlarken tek bir sayıya değil, **hangi chunk boyutunun neden**
  daha iyi/kötü çalıştığına da bakmaya çalış (raporunda bunu tartış).
