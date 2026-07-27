# Veri Sözlüğü — RAG Korpusu ve Gold Set

## `data/corpus/` — Doküman arşivi

- **27 adet `.txt` dosyası**, her biri bir Türkçe bilgi makalesi (Vikipedi tarzı, gerçek metin).
- **Dosya adı = konu.** Örnek: `Fotosentez.txt`, `DNA.txt`, `Kara_delik.txt`, `Osmanlı_İmparatorluğu.txt`.
- Toplam ~10.000 kelime. Makale uzunlukları değişkendir.
- Konu kümeleri kelime paylaşır (astronomi: `Güneş_Sistemi` / `Gezegen` / `Yıldız` / `Galaksi` / `Ay`; biyoloji: `Hücre` / `Virüs` / `Enzim` / `DNA` / `Bağışıklık_sistemi`). Bu, retrieval'ı zorlaştırır — ölçümü anlamlı kılan da budur.

## `data/gold_set.json` — Değerlendirme seti

27 soru içeren bir JSON listesi. Her eleman:

```json
{
  "soru": "Bitkilerin ışık enerjisini kimyasal enerjiye çevirdiği ... işlem nedir?",
  "dogru_dosya": "Fotosentez.txt",
  "anahtar_kelime": "klorofil"
}
```

| Alan | Anlamı |
|---|---|
| `soru` | Kullanıcının doğal dille sorduğu soru |
| `dogru_dosya` | Bu sorunun cevabının bulunduğu **doğru kaynak** (`corpus/` içindeki dosya adı) |
| `anahtar_kelime` | Doğru cevapta/kaynakta geçmesi beklenen bir kelime (citation / doğruluk kontrolü için kullanabilirsin) |

**Nasıl kullanılır:** Her soru için retrieval'ını çalıştır; getirdiğin ilk *k* kaynağın içinde `dogru_dosya` var mı diye bak. Bu oran (hit-rate), sistemin ne kadar doğru kaynağı bulduğunu gösterir.
