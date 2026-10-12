---
title: "هفته‌ی ۳: کلاس انتزاعی، dataclass، نوع‌دهی و خط لوله‌ی داده‌ی ترکیب‌پذیر"
subtitle: "یک دستورالعمل تکرارپذیر برای پاک‌سازی داده‌ی حسگر"
course: "هوش مصنوعی برای مهندسی هوافضا و مکانیک (دکتری)"
footer: "هفته‌ی ۰۳ | شیءگرایی: خط لوله‌های نوع‌دار و ترکیب‌پذیر"
language: fa
provenance: "محتوای جدید. مفاهیم پایتون با بازنویسی از W3Schools (ماژول abc، ماژول dataclasses، ماژول typing، __repr__، __eq__، تکرارگرها) و GeeksforGeeks (کلاس‌های انتزاعی، آشنایی با dataclass، جنریک‌ها در پایتون، str در برابر repr)؛ فیلتر Hampel برگرفته از Pearson, IEEE TCST 10(1), 2002؛ ضریب MAD از Rousseeuw & Croux, JASA 88, 1993."
---

<!-- Build: python tools/build_slides.py slides/fa/week03_data_pipeline_FA.md -->

## SLIDE 1 | یک لاگ، دو تحلیلگر، دو پاسخ
**Content:**
- گروه سازه می‌پرسد: بیشینه‌ی ضریب بار در پرواز آزمایشی هفته‌ی ۱ چقدر است؟
- تحلیلگر A: میانگین متحرک ۵ نمونه‌ای ← ۱٫۸۷g در ثانیه‌ی ۶۶٫۷
- تحلیلگر B: میانه‌ی متحرک و سپس میانگین ← ۱٫۸۰g در ثانیه‌ی ۱۹۲
- «بیشینه»ی A یک خطای حسگرِ پخش‌شده در پرواز افقی است
- پرسش: عدد داخل گزارش را کدام دستورالعمل ساخته است؟
**Visual:** IMAGE ../assets/week03/fig_two_analysts.png
**Notes:** بخش ۰ آزمایشگاه را زنده اجرا کنید. از کلاس بپرسید کدام عدد در گزارش می‌رود و خواننده‌ی گزارش چگونه می‌تواند بفهمد کدام اسکریپت آن را ساخته است. پرسش آخر را روی تخته بنویسید.

## SLIDE 2 | چرا اسکریپت، دستورالعمل نیست
**Content:**
- ترتیب مراحل در بدنه‌ی یک تابع پنهان است
- پارامترها (۵، ۷، center=True) همراه نتیجه ذخیره نمی‌شوند
- دو اسکریپت را نمی‌توان مقایسه کرد: analyst_a == analyst_b بی‌معناست
- نوع نادرست داده فقط هنگام خرابی کشف می‌شود، یا هرگز
**Visual:** دو جعبه‌ی اسکریپت «analyst_a.py» و «analyst_b.py» که به یک نمودار گزارش می‌رسند، با چهار علامت سؤال قرمز: ترتیب؟ پارامترها؟ یکسان؟ نوع‌ها؟
**Notes:** هر بند به یکی از ابزارهای این هفته می‌رسد: ترتیب ← tuple مراحل؛ پارامترها ← فیلدهای dataclass؛ مقایسه ← __eq__؛ نوع‌ها ← بررسی INPUT_TYPE و OUTPUT_TYPE.

## SLIDE 3 | مثال ۱: یک جهش در میانگین متحرک
**Content:**
- در پرواز افقی a_z ≈ ۱۰٫۳۵ m/s² (n ≈ ۱٫۰۵)؛ یک نمونه ۵۰٫۳۳ می‌خواند (۴۰+)
- میانگین ۵ نمونه‌ای اطراف آن: (۴ × ۱۰٫۳۵ + ۵۰٫۳۳) / ۵ = ۱۸٫۳۵ m/s²
- n = ۱۸٫۳۵ / ۹٫۸۰۷ = ۱٫۸۷g روی پنج نمونه‌ی پشت‌سرهم
- اول حذف جهش (میانه‌ی ۷ نمونه یک نمونه‌ی بد را نادیده می‌گیرد) ← ۱٫۸۰g، همان مانور کشش
- درس: پیش از هموارسازی، جهش‌ها را حذف کنید
**Visual:** دو ردیف ۷ جعبه‌ای: نمونه‌های خام a_z از ۱۰٫۲۷ تا ۱۰٫۳۸ با یک ۵۰٫۳۳ نارنجی؛ پس از میانگین ۵ نمونه‌ای، پنج جعبه حدود ۱۸٫۳ با رنگ نارنجی.
**Notes:** پیش از اجرای کد حساب را روی تخته انجام دهید (سطرهای ۳۳۳۱ تا ۳۳۳۷ لاگ، ثانیه‌ی ۶۶٫۶۲ تا ۶۶٫۷۴). نکته: فیلتری که میانگین می‌گیرد نمی‌تواند چیزی را که پخش کرده حذف کند.

## SLIDE 4 | کلاس پایه‌ی انتزاعی: قرارداد یک مرحله
**Content:**
- کلاس Stage(ABC) با @abstractmethod transform(data)
- اگر متدی پیاده نشود ← TypeError هنگام ساخت شیء، نه وسط اجرا
- ویژگی انتزاعی: @property همراه با @abstractmethod
- متدهای مشترک و غیرانتزاعی در خود ABC قرار می‌گیرند (__call__ نوع ورودی را بررسی می‌کند)
**Visual:** قطعه‌کد `class Stage(ABC): @property @abstractmethod def description(...)`، `@abstractmethod def transform(...)`، `def __call__(...)`؛ زیر آن `Forgetful()` ← پیام قرمز «TypeError: Can't instantiate abstract class Forgetful with abstract method description».
**Notes:** بازنویسی از GeeksforGeeks «Abstract Classes in Python» و W3Schools «Python abc Module». بخش ۱ آزمایشگاه، تمرین ۱٫۱ (PlausibleRange).

## SLIDE 5 | dataclass: پیکربندی همان کلاس است
**Content:**
- از فیلدهای دارای حاشیه‌نویسی نوع، __init__، __repr__ و __eq__ خودکار نوشته می‌شوند
- گزینه‌ی frozen=True: دستورالعمل‌های تغییرناپذیر و hash‌پذیر
- متد __post_init__: اعتبارسنجی (پنجره‌ی فرد و دست‌کم ۳) و یکسان‌سازی (list به tuple)
- در حالت frozen یک بار با object.__setattr__(self, "columns", tuple(...))
**Visual:** IMAGE ../assets/week03/fig_dataclass_generated.png
**Notes:** GeeksforGeeks «Understanding Python Dataclasses» و W3Schools «Python dataclasses Module». ابتدا نشان دهید ManualMovingAverage == ManualMovingAverage مقدار False می‌دهد.

## SLIDE 6 | مثال ۲: @dataclass چه نوشت؟
**Content:**
- چاپ Stage("smooth") می‌دهد Stage(name='smooth', window=5, tags=[])
- با default_factory=list هر شیء فهرست خودش را می‌گیرد
- نوشتن tags: list = [] ← ValueError همان لحظه‌ی تعریف کلاس
- با field(compare=False) مقدار یادگرفته‌شده‌ی mean_ در == نادیده گرفته می‌شود
- با eq و بدون frozen، __hash__ برابر None است ← hash‌ناپذیر
**Visual:** کد مسئله‌ی ۱ کلاس با چهار سطر خروجی در سمت راست؛ سطرهای ValueError و TypeError قرمز.
**Notes:** مسئله‌ی ۱ کلاس، ۵ دقیقه. default_factory را به خطای فهرست مشترک هفته‌ی ۲ ربط دهید.

## SLIDE 7 | حالت یادگرفته‌شده پیکربندی نیست
**Content:**
- مرحله‌ی Standardize در fit مقدارهای mean_ و std_ را یاد می‌گیرد
- با field(init=False, repr=False, compare=False)
- Standardize(["x"]) برازش‌شده و برازش‌نشده یک دستورالعمل‌اند
- فراخوانی transform پیش از fit ← NotFittedError
- متد get_params() فقط فیلدهای init=True را برمی‌گرداند
**Visual:** IMAGE ../assets/week03/fig_stage_hierarchy.png
**Notes:** همان قرارداد scikit-learn (زیرخط پایانی یعنی یادگرفته‌شده). در هفته‌ی ۶ با sklearn کار می‌کنیم؛ این اسلاید توضیح می‌دهد چرا رابط آن چنین است.

## SLIDE 8 | نوشتن دستی __repr__ و __eq__
**Content:**
- متد __repr__ برای برنامه‌نویس است؛ هدف: eval(repr(x)) == x
- متد __str__ برای گزارش است و print آن را ترجیح می‌دهد
- متد __eq__ برای نوع بیگانه NotImplemented برمی‌گرداند
- با تعریف __eq__، اگر __hash__ تعریف نشود None می‌شود
- دقیقاً همان چیزی را hash کنید که __eq__ مقایسه می‌کند، و فقط اگر هرگز تغییر نکند
**Visual:** قطعه‌کد کلاس Route با __repr__ ← "Route(['OIII', 'OIFM', 'OISS'])"، __str__ ← "OIII -> OIFM -> OISS"، __eq__ و __hash__؛ زیر آن سطر رفت‌وبرگشت که True چاپ می‌کند.
**Notes:** صفحه‌های W3Schools «__repr__()» و «__eq__()» و GeeksforGeeks «str() vs repr()». تمرین ۳٫۱ (Waypoint).

## SLIDE 9 | مثال ۳: فیلتر Hampel با دست
**Content:**
- ارتفاع‌ها h = ۱۰۰٫۰، ۱۰۰٫۴، ۹۹٫۸، ۱۰۶٫۰، ۱۰۰٫۳، ۱۰۰٫۱، ۹۹٫۹ متر؛ پنجره‌ی ۵
- برای k = ۳: میانه ۱۰۰٫۳؛ انحراف‌ها ۰٫۱، ۰٫۵، ۵٫۷، ۰٫۰، ۰٫۲ ← MAD = ۰٫۲
- آستانه ۳ × ۱٫۴۸۲۶ × ۰٫۲ = ۰٫۸۹ متر
- |۱۰۶٫۰ − ۱۰۰٫۳| = ۵٫۷ > ۰٫۸۹ ← جایگزینی با ۱۰۰٫۳
- برای k = ۴: |۱۰۰٫۳ − ۱۰۰٫۱| = ۰٫۲ ← حفظ
**Visual:** IMAGE ../assets/week03/fig_hampel_by_hand.png
**Notes:** ضریب ۱٫۴۸۲۶ = ۱/Φ⁻¹(۰٫۷۵) باعث می‌شود MAD برای نوفه‌ی گاوسی برآوردی از σ باشد (Rousseeuw & Croux 1993). بخش ۵ آزمایشگاه همین اعداد را با HampelFilter.outlier_mask تکرار می‌کند.

## SLIDE 10 | راهنمای نوع و جنریک‌ها
**Content:**
- راهنمای نوع قصد را مستند می‌کند؛ پایتون آن را اجبار نمی‌کند
- TypeVar یک نوع جانگهدار است و Generic[InT, OutT] یک کلاس نوع‌دار
- عبارت PipelineStage[DataFrame, ndarray] مثل list[int] خوانده می‌شود
- پارامترها در زمان اجرا پاک می‌شوند ← INPUT_TYPE و OUTPUT_TYPE را ذخیره کنید
- واژگان کد درس: ClassVar، Final و X | None
**Visual:** قطعه‌کد `InT = TypeVar("InT")`، `class TypedStage(ABC, Generic[InT, OutT])`، `class FrameToArray(TypedStage[pd.DataFrame, np.ndarray]): INPUT_TYPE = pd.DataFrame`؛ زیر آن `peak_g("9.8")` که فقط درون بدنه شکست می‌خورد.
**Notes:** GeeksforGeeks «Python Generics» و W3Schools «Python typing Module». ابزار mypy در محیط پین‌شده نیست؛ نامش را ببرید ولی نصبش نکنید.

## SLIDE 11 | ترکیب: مراحلی که با نوع به هم وصل می‌شوند
**Content:**
- عبارت a >> b شیء DataPipeline((a, b)) را می‌سازد
- سازنده شرط issubclass(left.OUTPUT_TYPE, right.INPUT_TYPE) را بررسی می‌کند
- ناسازگاری ← TypeError هنگام ساخت، پیش از ورود هر داده‌ای
- تغییرناپذیر: >> و برش (slice) خط لوله‌ی جدید برمی‌گردانند
- رابطه‌ی «دارد» (ترکیب)، نه «هست» (وراثت)
**Visual:** IMAGE ../assets/week03/fig_pipeline_chain.png
**Notes:** تمرین ۴٫۱ تابع can_follow() را می‌نویسد. مسئله‌ی ۴ کلاس می‌پرسد کدام خطاها هنگام ساخت و کدام هنگام اجرا رخ می‌دهند.

## SLIDE 12 | مثال ۴: آیا ترتیب مهم است؟
**Content:**
- سیگنال x = ۱، ۱، ۱، ۱۰، ۱، ۱، ۱؛ پنجره‌ی Hampel ۳ و پنجره‌ی میانگین ۳
- اول حذف جهش و بعد هموارسازی: در k = ۳ مقدار MAD صفر است ← ۱۰ به ۱؛ بیشینه ۱
- اول هموارسازی و بعد حذف جهش: ۱، ۱، ۴، ۴، ۴، ۱، ۱؛ هیچ نمونه‌ای پرت نیست؛ بیشینه ۴
- ترکیب جابه‌جاپذیر نیست: H∘S ≠ S∘H
- پس H >> S و S >> H باید نابرابر باشند
**Visual:** IMAGE ../assets/week03/fig_order_matters.png
**Notes:** مسئله‌ی ۳ کلاس. همان سازوکار ۱٫۸۷g در برابر ۱٫۸۰g در اسلاید آغازین.

## SLIDE 13 | کتابخانه‌ی درس: aerospace_ai.pipeline
**Content:**
- کلاس PipelineStage(ABC, Generic) با transform، fit، __call__ و registry
- مراحل frozen: SensorToFrame، SelectColumns، DropNonFinite، FillGaps
- و نیز HampelFilter، RollingMean، TimeDerivative و ToArray
- مرحله‌ی یادگیرنده: Standardize؛ ظرف: DataPipeline
- ۳۶ آزمون pytest در tests/test_pipeline.py
**Visual:** قطعه‌کد `recipe = SelectColumns([...]) >> HampelFilter(ACCEL, window=7) >> RollingMean(ACCEL, window=5)` و repr چندخطی آن.
**Notes:** مدرس: pipeline.py را پیش از کلاس بخوانید. در آن از __init_subclass__ برای registry، از typing.overload روی __getitem__ و از __slots__ در DataPipeline استفاده شده است.

## SLIDE 14 | خط لوله‌ای که گزارش می‌دهد و خودش را ذخیره می‌کند
**Content:**
- متد run() ← خروجی به‌علاوه‌ی یک StageReport برای هر مرحله
- تعداد سطر ورودی و خروجی، تعداد NaN، تعداد خانه‌های تغییرکرده و زمان
- متد to_config() ← دیکشنری ساده ← json.dumps
- DataPipeline.from_config(json) == original ← True
- فایل JSON را کنار هر عددی که گزارش می‌کنید ذخیره کنید
**Visual:** جدول گزارش خط لوله‌ی B روی لاگ پرواز (HampelFilter ۲۱۴۰ خانه و RollingMean ۴۴٬۸۱۵ خانه را تغییر داده‌اند) و تکه‌ای از JSON با "stage": "HampelFilter" و "window": 7.
**Notes:** registry که __init_subclass__ پر می‌کند نام‌های "stage" را به کلاس‌ها برمی‌گرداند؛ هفته‌ی ۴ این را الگوی Factory می‌نامد.

## SLIDE 15 | مثال ۵: هر دو تحلیلگر به‌شکل خط لوله
**Content:**
- خط لوله‌ی A = RollingMean(5) >> SelectColumns ← ۱٫۸۷g در ثانیه‌ی ۶۶٫۷
- خط لوله‌ی B = HampelFilter(7) >> RollingMean(5) >> SelectColumns ← ۱٫۸۰g در ثانیه‌ی ۱۹۲٫۰
- مقایسه‌ی A == B ← False، و repr دلیلش را نشان می‌دهد
- JSON پیکربندی B دقیقاً همان B را می‌سازد
**Visual:** IMAGE ../assets/week03/fig_two_analysts.png
**Notes:** بخش ۸ آزمایشگاه. با پنجره‌ی ۷، فیلتر Hampel حدود ۴٫۸٪ نمونه‌های شتاب‌سنج را جایگزین می‌کند: MAD کوچکِ حاصل از ۷ نمونه‌ی نوفه‌دار برخی نمونه‌های سالم را هم علامت می‌زند. اینجا بی‌ضرر است (به میانه‌ی محلی تبدیل می‌شوند)؛ بپرسید کدام پنجره نمونه‌های کمتری را علامت می‌زند.

## SLIDE 16 | برازش فقط روی داده‌ی آموزشی
**Content:**
- آموزش {۲، ۴، ۶} متر: μ = ۴ و σ = ۱٫۶۳۳ ← نمونه‌ی آزمون x = ۸: z = ۲٫۴۴۹
- برازش روی آموزش و آزمون با هم: μ = ۵ و σ = ۲٫۲۳۶ ← z = ۱٫۳۴۲
- نمونه‌ی آزمون عادی به نظر می‌رسد چون خودش «عادی» را تعریف کرده است
- متد fit خط لوله هر مرحله را روی خروجی مرحله‌ی قبل برازش می‌دهد
**Visual:** IMAGE ../assets/week03/fig_leakage.png
**Notes:** نشت داده موضوع اصلی هفته‌ی ۵ است؛ اینجا انگیزه‌ای است برای اینکه حالت یادگرفته‌شده در fit قرار گیرد و تمرین تمرینی ۳ داده را بر اساس زمان تقسیم کند.

## SLIDE 17 | هر حسگری وارد، آرایه‌ی آماده‌ی مدل خارج
**Content:**
- مرحله‌ی SensorToFrame هر BaseSensor را می‌پذیرد (چندریختی)
- IMU (±۴g) ← Hampel ← میانگین ← ToArray: آرایه‌ی (۱۴۹۳۷، ۳)، بیشینه ۱٫۸۰g
- حسگرهای LiDAR و GPS تمرین ۱ شما بدون تغییر وصل می‌شوند
- خط لوله هیچ کلاس حسگر مشخصی را import نمی‌کند
**Visual:** نمودار: IMUSensor، BarometerSensor، «GPSSensor شما» و «LidarSensor شما» همه با پیکان به یک جعبه‌ی SensorToFrame و سپس زنجیره.
**Notes:** حسگرهای مرجع LiDAR و GPS فقط پس از نمره‌دهی تمرین ۱ به src اضافه می‌شوند؛ خط لوله بر پایه‌ی BaseSensor نوشته شده و به آن‌ها نیاز ندارد. تمرین ۱۰٫۱.

## SLIDE 18 | مثال ۶: سطر، ثانیه نیست
**Content:**
- خط لوله‌ی نرخ صعود با پنجره‌ی ۵۱ سطری: بیشینه ۲۰٫۹۹ m/s در ثانیه‌ی ۹۹٫۷۶
- درست کنار یک وقفه‌ی ۱٫۲ ثانیه‌ای در ثبت (از ۹۹٫۹۸ تا ۱۰۱٫۲۰ ثانیه)
- پنجره در یک طرف زمان بیشتری را می‌پوشاند ← شیب کاذب
- ابتدا بازنمونه‌برداری روی شبکه‌ی یکنواخت ۱۰ هرتز ← ۱۹٫۹۱ m/s در ثانیه‌ی ۸۴٫۰
**Visual:** IMAGE ../assets/week03/fig_resample_gap.png
**Notes:** تمرین ۸٫۱ این اثر مصنوعی را پیدا می‌کند؛ وظیفه‌ی ۲ تمرین تمرینی ۳ مرحله‌ی Resample را می‌سازد که آن را حذف می‌کند.

## SLIDE 19 | آزمودن مراحل با pytest
**Content:**
- عددها را از مثال‌های دستی بیازمایید (Hampel در k = ۳ ← ۱۰۰٫۳)
- ردکردن‌ها را بیازمایید: پنجره‌ی بد، نوع نادرست، برازش‌نشده، ستون گم‌شده
- ناورداها را بیازمایید: ورودی دست‌نخورده، رفت‌وبرگشت repr و JSON
- ورودی‌های بد را با parametrize و قاب‌های کوچک را با fixture بسازید
**Visual:** قطعه‌کد `@pytest.mark.parametrize("window", [0, 1, 2, 4, -3])` ← `pytest.raises(ValueError, match="odd")`؛ خط ترمینال «77 passed».
**Notes:** فایل tests/test_pipeline.py الگوی وظیفه‌ی ۵ تمرین تمرینی ۳ است که روی سه پیاده‌سازی معیوب نمره داده می‌شود.

## SLIDE 20 | کارهای این هفته
**Content:**
- آزمایشگاه: notebooks/week03/week03_data_pipeline.ipynb (۶ تمرین TODO)
- سر کلاس: ۴ مسئله‌ی گرم‌کردن (خروجی dataclass، Hampel، ترتیب، خطای ساخت در برابر اجرا)
- تمرین تمرینی ۳: خط لوله‌ی داده‌های هوایی، حدود ۳٫۵ ساعت، تحویل هفته‌ی ۴
- مهلت تمرین ۱ ابتدای همین جلسه است
**Visual:** سه نماد: دفترچه، مداد، شاخه‌ی Git.
**Notes:** تمرین‌های تمرینی هفته‌هایی را پر می‌کنند که در سرفصل نقطه‌ی عطفی ندارند؛ شماره‌ی آن‌ها شماره‌ی هفته است، پس هرگز با A1 تا A6 تداخل ندارند.

## SLIDE 21 | آنچه اکنون می‌توانیم انجام دهیم
**Content:**
- قرارداد را با ABC و پیکربندی را با dataclass منجمد بنویسیم
- حالت یادگرفته‌شده را با field(init=False, repr=False, compare=False) جدا کنیم
- متدهای __repr__ و __eq__ و یک __hash__ سازگار بنویسیم
- با TypeVar و Generic کار کنیم و نوع را جایی که پایتون بررسی نمی‌کند بررسی کنیم
- مراحل نوع‌دار را در خط لوله‌ای ترکیب کنیم که گزارش می‌دهد و خودش را ذخیره می‌کند
**Visual:** دو اسکریپت اسلاید آغازین جای خود را به repr چاپ‌شده‌ی یک DataPipeline و نماد فایل JSON در کنار نمودار گزارش می‌دهند؛ تیک سبز روی ترتیب، پارامترها، برابری و نوع‌ها.
**Notes:** به پرسش آغازین برگردید: فایل JSON کنار نمودار پاسخ می‌دهد «کدام دستورالعمل این عدد را ساخته است». هفته‌ی بعد: الگوهای Strategy، Observer و Factory.
