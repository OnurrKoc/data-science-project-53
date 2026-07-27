# BilgiTR — Kaynaklı Soru-Cevap Asistanı (RAG)

**Bitirme Projesi · Veri Bilimi** • Tahmini süre: ~6-10 saat • Değerlendirme: **eğitmen incelemesi (manuel)**

---

## 📨 Durum

BilgiTR'de **data scientist** olarak çalışıyorsun. Elinizde bir doküman arşivi var (27 Türkçe bilgi makalesi). Ürün ekibinden şu istek geliyor:

> "Kullanıcılarımız arşivimizdeki bilgilere doğal dille soru sorabilsin istiyoruz: soruyu yazsınlar, sistem arşivden doğru bilgiyi bulup **kaynağını göstererek** cevaplasın. Ama dikkat: bana 'yaptım, çalışıyor' demen yetmez. Bu tür sistemler bazen alakasız doküman getirir. Senden **ne kadar doğru çalıştığını sayıyla kanıtlamanı** ve tasarım tercihlerini **ölçerek** savunmanı istiyorum."

Bu senin **bitirme projen**. Sana adım adım talimat vermiyoruz — bir RAG (Retrieval-Augmented Generation, getirim-destekli üretim) sistemini nasıl kuracağın ve **nasıl ölçeceğin** sana kalmış.

## 🎯 İş hedefi

Kullanıcı bir soru sorunca, doküman arşivinden **ilgili parçaları bulup kaynak göstererek** cevaplayan bir asistan kur. Ve — en önemlisi — bu sistemin **retrieval (bilgi getirme) kalitesini ölç ve raporla**.

## 📦 Elindeki veri

- `data/corpus/` — 27 Türkçe makale (`.txt`); dosya adı konuyu verir (ör. `Fotosentez.txt`).
- `data/gold_set.json` — 27 soru; her biri için **doğru kaynak dosya** ve cevapta geçmesi beklenen bir **anahtar kelime**. Bunu "ürün ekibinin sana verdiği test soruları" gibi düşün — sistemini bununla ölçeceksin.
- `data/veri_sozlugu.md` — veri ve gold set formatının açıklaması.

⚠️ Arşivde konu olarak yakın makaleler var (astronomi: Güneş Sistemi / Gezegen / Yıldız / Galaksi; biyoloji: DNA / Hücre / Virüs / Enzim). Bunlar kelime paylaştığı için retrieval'ı zorlar — ölçüm bu yüzden anlamlı.

## 🛠️ Başlarken

```bash
git clone <senin-fork-url>
cd data-science-project-53

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter lab                     # veya: jupyter notebook
```

## ✅ Bizden beklentiler

**Nasıl** çözeceğin serbest (hangi kütüphane, hangi retrieval yöntemi — TF-IDF, embedding, ne istersen). Ama iyi bir teslimatta şunları görmek istiyoruz:

1. **Retrieval hattı kur** — dokümanları parçalara (chunk) böl, indeksle, bir soruya en ilgili parçaları getir.
2. **Kaynak göster (citation)** — cevabın hangi doküman(lar)dan geldiğini döndür. Kaynaksız cevap bu projede eksik sayılır.
3. **Ölç** — `gold_set.json` üzerinde retrieval kalitesini bir metrikle raporla (ör. **hit-rate**: doğru kaynak, getirilen ilk *k* sonuç içinde mi). Sayı ver.
4. **Karşılaştır (en az bir tasarım kararı)** — parça boyutu, getirilen sonuç sayısı *k*, ya da yöntem gibi bir tercihi değiştir ve etkisini **sayıyla** göster (ablation). **Neden** öyle çıktığını yorumla — bu projenin özü "kur"maktan çok "kur + ölç + karşılaştır".
5. **(Opsiyonel) Cevap üret** — getirilen parçalardan bir cevap oluştur. Gerçek bir LLM **gerekmez**: API key istemeyen basit/mock bir üretici yeterli. İstersen `.env.example`'ı kullanıp kendi anahtarınla gerçek LLM de deneyebilirsin (odak retrieval + ölçüm).

## 📤 Teslim edeceklerin

| Dosya | Ne olmalı |
|---|---|
| `analiz.ipynb` | Uçtan uca defterin: korpusu yükleme → parçalama → indeksleme → getirme → ölçüm → karşılaştırma, kararlarını **anlatarak**. |
| `rag.py` | Yeniden çalıştırılabilir script: korpusu indeksler, `gold_set` üzerinde metriği hesaplayıp ekrana basar. (Biz `python rag.py` diyip çalıştırabilmeliyiz.) — yani notebook'ta kurduğun hattın temiz, tek komutla çalışan hali. |
| `RAPOR.md` | İş diliyle raporun (`RAPOR_SABLONU.md`'yi doldur). |

## 🧭 Nasıl değerlendirilecek

Otomatik test yok — bir eğitmen aşağıdaki rubriğe göre değerlendirecek:

| Boyut | Puan |
|---|---:|
| Retrieval hattı & kaynak gösterimi (citation) | 25 |
| Ölçüm & metrik doğruluğu (gold set üzerinde) | 25 |
| Karşılaştırma / ablation & yorumu | 20 |
| Veri işleme & parçalama (chunking) | 10 |
| Rapor & iş içgörüsü | 15 |
| Kod kalitesi & tekrar çalıştırılabilirlik (`rag.py`) | 5 |
| **Toplam** | **100** |

Geçmek için ~70/100 hedefle. **Puanın çoğu bir RAG kurmakta değil; onu ölçmen ve tasarım kararlarını sayıyla savunmanda.**

## 📈 Başarı hedefi

Katı bir eşik yok. `gold_set` üzerinde **anlamlı bir hit-rate** hedefle (yön verelim: iyi bir çözüm ~0.7 ve üzerini yakalar). Ama asıl değerlendirilen: sonucu **nasıl ölçtüğün** ve karşılaştırmandan **ne çıkardığın**.

## 🚀 Nasıl gönderirsin

1. Repoyu **fork'la**, çöz, repo'nu **public** bırak.
2. Kaizu'da bu projede **"İncelet 🔍"** butonuna bas.
3. **GitHub repo linkini** ve **neler yaptığını** (yaklaşımın, ölçüm sonuçların, bakmamızı istediğin yerler) yaz, gönder.
4. Eğitmenin inceleyecek — yanıt **2-3 iş günü** sürebilir; sonucu Kaizu'da göreceksin. Takılırsan eğitmenine danışabilirsin.

> Not: Bu projenin onayı, sıradaki bitirme projeni açar.

## 💡 Hatırlatmalar

- Cevap üretimi için **API key şart değil** — puanlama retrieval + ölçüm üzerinden.
- Kütüphane seçimi sana ait; `requirements.txt` bir başlangıç seti.
- Amaç kusursuz bir sistem değil; **ölçen ve kararını savunan** bir mühendis gibi çalışman.

Başarılar 🚀
