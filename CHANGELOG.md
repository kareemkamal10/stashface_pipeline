# Changelog

كل تعديل بيتم على المشروع بيتسجل هنا، الأحدث فوق.

## 2026-09-03

### إعادة هيكلة ناتج الاستعلام: مجموعات بدل أزواج
- شيل `confirmed_duplicates.json` و`needs_human_review.json` نهائيًا
- بدالهم ملف واحد: `duplicate_groups.json`
- كل عنصر فيه مجموعة (`members`) تمثل شخص واحد متكرر عبر أكتر من `id`:
  - كل عضو: `id`, `file_type`, `image`, `cosine_score`, `need_review`
  - الأعضاء متسجلين مرتبين تنازليًا حسب `cosine_score` (الأعلى فوق، الأقل تحت)
  - الـ `cosine_score` بيتحسب بالنسبة لـ "anchor" المجموعة (العضو الأكتر تمثيلاً لباقي الأعضاء، مش أول عضو دخل بالصدفة)
  - `need_review = true` لو `cosine_score < 70`، وإلا `false`
- التجميع بقى بيستخدم Union-Find على top-10 لكل عنصر (مش أقرب نتيجة واحدة بس) — عشان يمسك حالات الشخص المتكرر 3 مرات أو أكتر مش بس زوج بزوج
- الأشخاص اللي مالهمش أي تكرار (مجموعة من عنصر واحد) بيتشالوا من التقرير خالص

### نقل مكان الكود من /kaggle/working لـ /kaggle/temp
- خلية الـ `git clone` في `stashface_kaggle.ipynb` بقت بتحط الكود في `/kaggle/temp/stashface_pipeline` بدل `/kaggle/working/stashface_pipeline`
- كده كل حاجة (الكود + بيانات الموديل + التقارير) بقت تحت `/kaggle/temp`، من غير أي تخزين على `/kaggle/working` خالص

### إنشاء dedupe_pipeline.py + تحديث الـ notebook
- سكريبت جديد بيبني قاعدة بيانات وجوه مؤقتة (في الرام) من ملفي `performers_with_tpdb.json` و`performers_without_tpdb.json` مع بعض، من غير أي اعتماد على `DataManager`/`performers.zvec` القديمين
- بيستخدم نفس الـ detection/embedding القديمين (`buffalo_l` + AdaFace ViT-B)
- تصنيف فشل واضح لكل حالة: `download_failed.json`, `no_face_detected.json`, `multiple_faces_detected.json` — صور فيها أكتر من وش تتعتبر تالفة ومش بتتستخدم
- بناء + استعلام resumable وبـ batching (نفس اتفاقيات المشروع: checkpoint بعد كل batch، سطر ملخص واحد بس، تحميل صور متوازي)
- الاستعلام بيعمل top-10 لكل عنصر، مش top-1 — عشان يقدر يمسك أكتر من نسخة لنفس الشخص
- في الآخر بيرفع كل حاجة أوتوماتيك لمجلد `reports/` جوا نفس الـ HF dataset repo، من غير خلية رفع يدوي منفصلة
- الـ `stashface_kaggle.ipynb` اتعدل يقرا الملفين من نفس الـ dataset، يشغل `dedupe_pipeline.py` بدل `run_kaggle.py`، ويطبع ملخص محلي بسيط في الآخر
