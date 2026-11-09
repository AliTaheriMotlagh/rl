// Training texts for the language-model lessons.
// Classical Persian poems (public domain) + generated texts with simple rules,
// so you can see a model discover the rules by itself.

function makeFaSentences(n = 260) {
  const subj = ["علی", "مریم", "سارا", "رضا", "پدرم", "مادرم", "دوستم", "معلم", "بچه‌ها", "گربه"];
  const plural = { "بچه‌ها": true };
  const obj = ["کتاب", "نان", "آب", "سیب", "نامه", "غذا", "چای", "گل"];
  const place = ["مدرسه", "خانه", "بازار", "پارک", "کتابخانه", "دریا"];
  const out = [];
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  for (let i = 0; i < n; i++) {
    const s = pick(subj), pl = plural[s];
    const t = Math.random();
    if (t < 0.35) out.push(`${s} به ${pick(place)} ${pl ? "رفتند" : "رفت"}.`);
    else if (t < 0.7) out.push(`${s} ${pick(obj)} ${pl ? "خوردند" : "خورد"}.`);
    else out.push(`${s} ${pick(obj)} را از ${pick(place)} ${pl ? "آوردند" : "آورد"}.`);
  }
  return out.join("\n") + "\n";
}

function makeEnSentences(n = 260) {
  const subj = ["the cat", "the dog", "a bird", "my friend", "the teacher", "a child"];
  const verb = ["sees", "likes", "finds", "wants", "eats"];
  const obj = ["the ball", "an apple", "the book", "a fish", "the cake", "some water"];
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  const out = [];
  for (let i = 0; i < n; i++) out.push(`${pick(subj)} ${pick(verb)} ${pick(obj)}.`);
  return out.join("\n") + "\n";
}

function makeMath() {
  const lines = [];
  for (let r = 0; r < 4; r++)
    for (let a = 0; a < 10; a++) for (let b = 0; b < 10; b++) lines.push(`${a}+${b}=${a + b}`);
  for (let i = lines.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [lines[i], lines[j]] = [lines[j], lines[i]];
  }
  return lines.join("\n") + "\n";
}

const POEMS = `بنی آدم اعضای یکدیگرند
که در آفرینش ز یک گوهرند
چو عضوی به درد آورد روزگار
دگر عضوها را نماند قرار
تو کز محنت دیگران بی غمی
نشاید که نامت نهند آدمی
توانا بود هر که دانا بود
ز دانش دل پیر برنا بود
چو ایران نباشد تن من مباد
بدین بوم و بر زنده یک تن مباد
الا یا ایها الساقی ادر کاسا و ناولها
که عشق آسان نمود اول ولی افتاد مشکل‌ها
به بوی نافه‌ای کاخر صبا زان طره بگشاید
ز تاب جعد مشکینش چه خون افتاد در دل‌ها
این قافله عمر عجب می‌گذرد
دریاب دمی که با طرب می‌گذرد
ساقی غم فردای حریفان چه خوری
پیش آر پیاله را که شب می‌گذرد
`;

const CORPORA = {
  fa: { name: "جمله‌های سادهٔ فارسی (ساخته‌شده با قاعده)", make: () => makeFaSentences() },
  math: { name: "جمع اعداد یک‌رقمی (مثل 3+4=7)", make: makeMath },
  en: { name: "English simple sentences", make: () => makeEnSentences() },
  poems: { name: "شعر کلاسیک فارسی (سعدی، فردوسی، حافظ، خیام)", make: () => POEMS.repeat(3) },
};

if (typeof module !== "undefined") module.exports = { CORPORA };
