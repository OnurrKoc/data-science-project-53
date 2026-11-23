# RAG Projesi — Rapor

> Bu şablonu doldur ve `RAPOR.md` olarak kaydet. Teknik olmayan bir yöneticinin de
> anlayacağı **iş diliyle** yaz. Netlik önemli, uzunluk değil. Başlıkları silme.

## 1. Özet (3-4 cümle)
BilgiTR'nin 27 Türkçe bilgi makalesinden oluşan arşivi için sorulan sorulara doğru kaynak dokümanı göstererek (citation) yanıt veren bir RAG (Retrieval-Augmented Generation) sistemi kuruldu. 27 soruluk test seti (`gold_set.json`) üzerinde yapılan ölçümlerde sistemimiz ilk 3 sonuçta (**Hit-Rate@3**) **%96.3 (26/27)** doğruluk ve **0.926 MRR** skoru yakalayarak hedef eşiği (0.70) fazlasıyla aştı. En önemli bulgumuz, Türkçe sondan eklemeli bir dil olduğu için karakter tabanlı n-gram yönteminin standart kelime aramasına göre anahtar kelime yakalama oranını (%48.1'den %55.6'ya) belirgin şekilde artırması ve `k=3` yapısının en dengeli kullanıcı deneyimini sunması oldu.

## 2. Sistemi nasıl kurdun
Dokümanları bağlam bütünlüğünü koruyacak şekilde **100 kelimelik** ve ardışık parçalar arasında **20 kelime örtüşmeli (overlap)** 134 parçaya (chunk) böldük. Her parçanın başına ilgili makalenin konu başlığını ekleyip Türkçe karakterlere (`İ -> i`, `I -> ı`) duyarlı ön işlemeden geçirdik. Kullanıcı sorusu geldiğinde hem soruyu hem de doküman parçalarını **Karakter N-Gram (3–5) TF-IDF** vektörlerine dönüştürüyor, kosinüs benzerliği (Cosine Similarity) ile en yüksek skorlu ilk 3 parçayı kaynak dosya adıyla birlikte kullanıcıya sunuyoruz.

## 3. Ölçüm
Sistemin başarısını `gold_set.json` üzerindeki 27 soru ile üç farklı metrikte ölçtük:
* **Hit-Rate@3 = 0.963 (26/27):** Kullanıcıların sorduğu her 27 sorudan 26'sında doğru kaynak makale ilk 3 sonuç arasında yer aldı.
* **MRR@3 = 0.926:** Doğru kaynakların ezici çoğunluğu doğrudan **1. sırada** kullanıcıya sunuldu (tek sonuçta dahi **%88.9** isabet sağlandı).
* **Anahtar Kelime İsabeti = 0.556 (15/27):** Getirilen 100 kelimelik ilk 3 parçanın içinde beklenen spesifik terimin doğrudan geçme oranıdır; parçalar makale sonundaki detayları içermediğinde dahi doğru kaynak dosya bulunabilmektedir.

## 4. Karşılaştırma (ablation)
Tasarım kararlarımızı kanıtlamak için 7 farklı konfigürasyonu test ettik:

| Deney | Parça Sayısı | Hit-Rate@k | MRR | Keyword Hit | İsabet |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline (Kelime TF-IDF, chunk=100, k=3) | 134 | 0.963 | 0.938 | 0.481 | 26/27 |
| 2. Karakter N-Gram (chunk=100, k=3) | 134 | 0.963 | 0.926 | 0.556 | 26/27 |
| 3. Küçük Parça (char_ngram, chunk=50, k=3) | 256 | 0.963 | 0.944 | 0.444 | 26/27 |
| 4. Büyük Parça (char_ngram, chunk=250, k=3) | 54 | 0.926 | 0.864 | 0.852 | 25/27 |
| 5. Tam Doküman (Bölmeden, chunk=2000, k=3) | 27 | 1.000 | 0.901 | 1.000 | 27/27 |
| 6. Tek Sonuç (char_ngram, chunk=100, k=1) | 134 | 0.889 | 0.889 | 0.481 | 24/27 |
| 7. Geniş Sonuç (char_ngram, chunk=100, k=5) | 134 | 0.963 | 0.926 | 0.704 | 26/27 |

* **Neden Karakter N-Gram ve `chunk=100, k=3` seçildi?** Türkçe eklemeli bir dil olduğu için karakter n-gramları ek almış kelimeleri kökleriyle eşleştirerek Keyword Hit oranını artırdı. Makaleler kısa (~370 kelime) olduğu için Tam Doküman %100 isabet verse de uzun metin birikimi nedeniyle sıralama keskinliğini (`MRR: 0.901`) düşürdü. Gerçek dünya ölçeklenebilirliği, LLM bağlam penceresi verimliliği ve yüksek sıralama keskinliği (`MRR: 0.926`) nedeniyle `chunk=100` ana mimari olarak seçildi. `k=1`'den `k=3`'e çıkmak isabeti %88.9'dan %96.3'e taşırken, `k=5` ek bir dosya isabeti sağlamayıp gürültüyü artırdığı için `k=3` tercih edildi.

## 5. Sistem nerede zorlanıyor?
Sistem özellikle **aynı konu kümesinde yer alan ve ortak kelime havuzu paylaşan makalelerde** zorlanmaktadır. Örneğin Astronomi (`Yıldız`, `Gezegen`, `Güneş Sistemi`, `Galaksi`, `Ay`) ve Biyoloji (`Hücre`, `Virüs`, `Enzim`, `DNA`, `Bağışıklık sistemi`) konularında soru bir gezegeni veya uyduyu sorsa dahi soru cümlesinde geçen *"yıldız"* veya *"kütleçekimi"* gibi ortak kelimeler nedeniyle ilgili diğer makaleler yüksek benzerlik skoru alarak üst sıralara çıkabilmektedir.

## 6. Tavsiyen
* Sistemi canlıya alırken kullanıcıya tek bir sonuç yerine **en yüksek skorlu ilk 3 kaynağı (`k=3`)** göstermek doğru bilgiye ulaşma oranını %96'nın üzerine çıkaracaktır.
* Arşivde cevabı olmayan sorulara alakasız doküman döndürmemek (halüsinasyon riskini sıfırlamak) için kosinüs benzerlik skoru belirli bir eşiğin (ör. `0.15`) altında kalan sorgularda *"Arşivde bu soruyla eşleşen güvenilir bir kaynak bulunamadı"* uyarısı verilmelidir.


## 7. (Opsiyonel) Daha fazla zamanın olsa
Daha fazla zamanım olsaydı, kelimeler birebir eşleşmese bile sorunun anlamını anlayan (örneğin "gökyüzündeki ışık küresi" sorusunu "yıldız" makalesiyle eşleştirebilen) yapay zeka tabanlı arama modellerini denerdim. Ayrıca getirilen doğru parçalardan kullanıcıya doğrudan akıcı bir özet cümle yazan bir dil modeli (LLM) eklerdim.