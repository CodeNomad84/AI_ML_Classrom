---
title: "هفته‌ی ۲: کلاس، __init__ و وراثت"
subtitle: "یک رابط مشترک برای همه‌ی حسگرهای وسیله"
course: "هوش مصنوعی برای مهندسی هوافضا و مکانیک (دکتری)"
footer: "هفته‌ی ۰۲ | شیءگرایی: سلسله‌مراتب حسگرها"
language: fa
provenance: "محتوای جدید. مفاهیم شیءگرایی با بازنویسی از W3Schools (کلاس‌ها، __init__، ویژگی‌های کلاس، وراثت، چندریختی) و GeeksforGeeks (متغیر کلاس و نمونه، super، property، کلاس‌های انتزاعی، dataclass، شروع کار با Pytest)؛ مدل ISA برگرفته از Anderson, Introduction to Flight, Ch. 3؛ دستگاه محلی زمین تخت برگرفته از Groves, Principles of GNSS, Ch. 2."
---

<!-- Build: python tools/build_slides.py slides/fa/week02_oop_sensor_hierarchy_FA.md -->

## SLIDE 1 | سه حسگر، یک ربات
**Content:**
- IMU: شش عدد با نرخ ۵۰ هرتز، دارای بایاس و امکان اشباع
- LiDAR: شانزده برد با نرخ ۱۰ هرتز؛ بدون بازتاب، ۰٫۰ یا بی‌نهایت
- GPS: عرض، طول، ارتفاع و HDOP با نرخ ۵ هرتز؛ برخی موقعیت‌ها خراب‌اند
- همه‌ی ماژول‌های بعدی (خط، مانع، کنترل) از این داده‌ها تغذیه می‌کنند
- پرسش: چگونه حسگر چهارم را بدون ویرایش همه‌چیز اضافه کنیم؟
**Visual:** نمای بالای یک ربات زمینی کوچک با سه حسگر برچسب‌دار (IMU در بدنه، LiDAR روی سقف با حلقه‌ی پرتوها، آنتن GPS)؛ از هر کدام پیکانی به قالب داده‌ی خام متفاوت.
**Notes:** جلسه را با وسیله شروع کنید، نه با پایتون. از کلاس بپرسید کد سه حسگر متفاوت را چگونه سازمان می‌دهند. پرسش آخر را روی تخته بنویسید و تا پایان نگه دارید.

## SLIDE 2 | چرا نسخه‌ی رویه‌ای می‌شکند
**Content:**
- برای هر کار یک تابع و درون هر کدام زنجیره‌ی if/elif روی نوع حسگر
- افزودن بارومتر یعنی ویرایش validate، preprocess، to_frame و plot
- پیکربندی (بایاس‌ها) به متغیر سراسری می‌رود
- اشتباه تایپی بی‌صدا از زنجیره عبور می‌کند
**Visual:** IMAGE ../assets/week02/fig_procedural_vs_oop.png
**Notes:** جدول چپ: ستون جدید به همه‌ی توابع دست می‌زند. جدول راست: حسگر جدید فقط دو متدی را می‌نویسد که واقعاً متفاوت‌اند و بقیه را به ارث می‌برد.

## SLIDE 3 | مثال ۱: اشتباه تایپی بی‌صدا
**Content:**
- فراخوانی preprocess("imu", raw) بایاس را حذف می‌کند
- فراخوانی preprocess("IMU", raw) داده‌ی خام را برمی‌گرداند؛ بدون خطا
- فراخوانی preprocess("imu", [0.35, -0.05]) دو مقدار را می‌پذیرد؛ zip زود متوقف می‌شود
- داده‌ی بایاس‌دار بدون اینکه کسی بفهمد وارد مدل می‌شود
**Visual:** قطعه‌کد `def preprocess(sensor_type, data): if sensor_type == "imu": ... elif "lidar": ... return data` همراه با سه فراخوانی و خروجی آن‌ها؛ دو خروجی نادرست با رنگ نارنجی.
**Notes:** بخش ۰ آزمایشگاه را زنده اجرا کنید. نکته: خطاهایی که استثنا ایجاد نمی‌کنند گران‌ترین خطاها هستند.

## SLIDE 4 | کلاس، شیء، __init__ و self
**Content:**
- کلاس نقشه است و شیء یک نمونه‌ی ساخته‌شده از آن
- متد __init__ هنگام ساخت اجرا می‌شود و ویژگی‌ها را روی self ذخیره می‌کند
- متد تابعی درون کلاس است؛ پایتون خود شیء را به‌عنوان self می‌فرستد
- دو شیء، یک کلاس، وضعیت متفاوت (دو تنظیم QNH)
**Visual:** کد `class SimpleBarometer: def __init__(self, sensor_id, reference_pressure_pa=101325.0): ... def altitude_m(self, p): ...` کنار دو جعبه‌ی شیء «baro0 (101 325 Pa)» و «baro_qnh (100 500 Pa)».
**Notes:** بازنویسی از W3Schools («Python Classes» و «__init__() Method»). تأکید کنید که self فقط نخستین پارامتر است و نامش یک قرارداد است.

## SLIDE 5 | ویژگی کلاس در برابر ویژگی نمونه
**Content:**
- تعریف در بدنه‌ی کلاس: میان همه‌ی اشیا مشترک است (مناسب ثابت‌ها)
- تعریف با self در __init__: هر شیء نسخه‌ی خودش را دارد (وضعیت)
- تغییر یک لیست مشترک داده‌ها را میان حسگرها نشت می‌دهد
- انتساب روی نمونه سایه می‌اندازد، اما تغییر درجا (mutation) نه
**Visual:** IMAGE ../assets/week02/fig_shared_list.png
**Notes:** برگرفته از GeeksforGeeks «Class (static) and instance variables». مسئله‌ی ۱ کلاسی همین را تمرین می‌کند. قاعده: وضعیت تغییرپذیر در __init__ ساخته شود.

## SLIDE 6 | مثال ۲: ارتفاع ISA با دست
**Content:**
- رابطه: h = (T0/L)[1 − (p/p_ref)^(RL/g0)]؛ T0/L = 44 330.8 m و RL/g0 = 0.190263
- برای p = 95 000 Pa: نسبت 0.937577 و لگاریتم آن −0.064457
- توان: e^(−0.012264) = 0.987811
- ارتفاع: h = 44 330.8 × 0.012189 = 540.3 m
- با QNH = 100 500 Pa: ۴۷۲٫۲ متر (همان کلاس، شیء دیگر)
**Visual:** IMAGE ../assets/week02/fig_isa_altitude.png
**Notes:** ابتدا محاسبه را روی تخته انجام دهید و سپس SimpleBarometer را اجرا کنید. تمرین ۱٫۱ همان حالت QNH است.

## SLIDE 7 | ویژگی (property): صفتی که خودش را بررسی می‌کند
**Content:**
- دکوراتور @property: شبیه صفت است اما متد اجرا می‌کند
- setter اعتبارسنجی می‌کند: rate_hz = 0 خطای ValueError می‌دهد
- بدون setter یعنی فقط‌خواندنی (sensor_id)
- مقادیر مشتق‌شده سازگار می‌مانند: period_s = 1 / rate_hz
- پیشوند _ یعنی «داخلی است، دست نزنید»
**Visual:** کد `@property def rate_hz(self)` و `@rate_hz.setter def rate_hz(self, value): if not value > 0: raise ValueError`؛ پیکانی از `imu.rate_hz = 0` به «ValueError» قرمز.
**Notes:** برگرفته از GeeksforGeeks «Python property() function». بپرسید چرا period_s را در __init__ ذخیره نکنیم؟ چون با تغییر نرخ کهنه می‌شود.

## SLIDE 8 | وراثت و super()
**Content:**
- با class Barometer(Sensor) همه‌ی متدهای Sensor به ارث می‌رسد
- بازنویسی (override): همان نام متد، بدنه‌ی مخصوص حسگر
- فراخوانی super().__init__(...) بخش والد را مقداردهی می‌کند
- عبارت isinstance(baro, Sensor) درست است
- MRO: ترتیبی که پایتون برای یافتن متد جست‌وجو می‌کند
**Visual:** دو جعبه‌ی روی هم: Sensor (sensor_id، rate_hz، readings، add_reading، preprocess) و Barometer (reference_pressure_pa، بازنویسی preprocess) با پیکان وراثت.
**Notes:** برگرفته از W3Schools «Python Inheritance» و GeeksforGeeks «super()». به‌صورت زنده Barometer.__mro__ را چاپ کنید.

## SLIDE 9 | مثال ۳: دنبال‌کردن super() در سه کلاس
**Content:**
- زنجیره: IMU ← InertialSensor ← Sensor ← object
- هر __init__ ابتدا چاپ می‌کند و سپس super().__init__ را صدا می‌زند
- خروجی: IMU init، Inertial init، Sensor init
- خروجی describe(): "IMU/inertial imu0 @ 200 Hz"
- فراموشی یک super().__init__: خطای AttributeError خیلی دیرتر
**Visual:** زنجیره‌ی عمودی چهار جعبه (IMU، InertialSensor، Sensor، object)؛ پیکان‌های رو به پایین «super().__init__» و رو به بالا «مقدار بازگشتی describe()».
**Notes:** مسئله‌ی ۲ کلاسی. دانشجویان پیش از اجرا پیش‌بینی کنند. نکته‌ی اصلی، خطای دیرهنگام AttributeError است.

## SLIDE 10 | کلاس پایه‌ی انتزاعی: رابط به‌عنوان قرارداد
**Content:**
- کلاس BaseSensor(ABC) با متدهای انتزاعی validate و preprocess
- ساختن BaseSensor(...) خطای TypeError می‌دهد
- زیرکلاسی که متدی را جا بیندازد (یا غلط بنویسد) هم TypeError می‌دهد
- خطا هنگام ساخت شیء، نه وسط پرواز
**Visual:** کد `class ForgetfulBarometer(AbstractSensor): def pre_process(...)` و خطای حاصل `TypeError: Can't instantiate abstract class ... with abstract method preprocess`.
**Notes:** برگرفته از GeeksforGeeks «Abstract Classes in Python». هفته‌ی ۳ عمیق‌تر می‌شود (ویژگی‌های انتزاعی، نوع‌دهی).

## SLIDE 11 | dataclass: رکورد SensorReading
**Content:**
- دکوراتور @dataclass متدهای __init__، __repr__ و __eq__ را از فیلدها می‌سازد
- frozen=True: اندازه‌گیری ثبت‌شده تغییر نمی‌کند (و hash‌پذیر است)
- slots=True: بدون __dict__ برای هر شیء و حافظه‌ی کمتر
- مقادیر tuple هستند؛ آرایه == و hash را خراب می‌کند
**Visual:** مقایسه‌ی کنار هم: کلاس دستی ۱۵ خطی ManualReading و نسخه‌ی ۶ خطی `@dataclass(frozen=True, slots=True) class SensorReading: timestamp: float; values: tuple[float, ...]; sensor_id: str; is_valid: bool = True`.
**Notes:** برگرفته از GeeksforGeeks «Understanding Python Dataclasses». خطای FrozenInstanceError را زنده نشان دهید.

## SLIDE 12 | مثال ۴: بایاس و اشباع IMU با دست
**Content:**
- IMU با بازه‌ی ±۴g: حد ۳۹٫۲۳ m/s²؛ بایاس شتاب (0.10, −0.05, 0.20) و ژیرو (0.8, −0.3, 0.1)
- نمونه‌ی (0.35, −0.05, 10.05 | 0.8, −0.3, 2.1) ← (0.25, 0, 9.85 | 0, 0, 2.0)؛ n = 1.005
- مقدار NaN در a_y ← نامعتبر، به‌صورت خام ذخیره می‌شود
- a_z = 39.50 ≥ 39.23 ← اشباع و نامعتبر؛ مقدار واقعی نامعلوم است
- نمونه‌ی (1.30, −0.45, 19.80 | ...) ← (1.2, −0.4, 19.6)؛ n = 2.003
**Visual:** جدول چهار سطری: نمونه‌ی خام، معتبر؟، مقادیر پیش‌پردازش‌شده، ضریب بار؛ سطرهای نامعتبر با رنگ نارنجی.
**Notes:** همان اعداد بخش ۷ آزمایشگاه و آزمون‌ها. نمونه‌ی آخر همان مثال حل‌شده‌ی هفته‌ی ۱ است؛ پیوستگی برای دانشجویان کم‌تجربه مهم است.

## SLIDE 13 | سلسله‌مراتب حسگرهای درس
**Content:**
- BaseSensor بافر، بررسی زمان، آرایه و DataFrame را بر عهده دارد
- زیرکلاس‌ها فقط validate() و preprocess() را می‌نویسند
- IMUSensor و BarometerSensor در src/aerospace_ai/sensors.py
- LidarSensor و GPSSensor: تمرین ۱
**Visual:** IMAGE ../assets/week02/fig_class_hierarchy.png
**Notes:** مدرس: sensors.py را پیش از کلاس بخوانید؛ از __slots__، __init_subclass__ و بافر deque هم استفاده می‌کند که در هفته‌های ۳ و ۴ بحث می‌شوند.

## SLIDE 14 | الگوی متد قالب (template method): add_reading
**Content:**
- الگوریتم یک بار در کلاس پایه ثابت شده است
- گام‌های ۳ و ۴ به زیرکلاس سپرده می‌شوند
- نمونه‌های نامعتبر خام و با is_valid=False نگه داشته می‌شوند
- to_array() فقط نمونه‌های معتبر را برمی‌گرداند
**Visual:** IMAGE ../assets/week02/fig_template_method.png
**Notes:** نام الگو را همین حالا بگویید؛ هفته‌ی ۴ الگوهای Strategy، Observer و Factory را پوشش می‌دهد.

## SLIDE 15 | مثال ۵: لاگ هفته‌ی ۱ از درون IMUSensor
**Content:**
- کالیبره‌کردن بایاس‌ها از بازه‌ی t < 25 s (توقف و تراز)
- ورود ۱۴٬۹۴۰ نمونه با بازه‌ی ±۴g
- ۳ خار (حدود ۵۰ m/s²) به‌عنوان اشباع رد می‌شوند
- بیشینه‌ی ضریب بار ۱٫۸۱g؛ زیر حد ۲٫۰g
- هفته‌ی ۱ میانه‌ی متحرک لازم داشت؛ اینجا شیء حدود خودش را می‌شناسد
**Visual:** IMAGE ../assets/week02/fig_imu_validation.png
**Notes:** بخش ۹ آزمایشگاه. هشدار: رد کردن اینجا کار می‌کند چون خارها از حد بازه بیشترند؛ خارهای درون بازه همچنان فیلتر لازم دارند.

## SLIDE 16 | چندریختی: یک حلقه، حسگرهای متعدد
**Content:**
- for sensor in suite: sensor.add_reading(t, raw)
- هیچ if/elif روی نوع حسگر وجود ندارد
- ThermometerSensor جدید بدون هیچ تغییری در حلقه اضافه می‌شود
- فراخوانی یکسان، رفتار مخصوص هر نوع
**Visual:** حلقه‌ی کد روی `[IMUSensor, BarometerSensor, ThermometerSensor]` که شناسه، کلاس، کانال‌ها، تعداد نمونه و نمونه‌های ردشده را چاپ می‌کند.
**Notes:** برگرفته از W3Schools «Python Polymorphism». تمرین ۸٫۱ دماسنج را می‌سازد (ورودی کلوین، خروجی سلسیوس).

## SLIDE 17 | LiDAR: پرتوها، زاویه‌ها و بازتاب‌های گمشده
**Content:**
- پرتو i در زاویه‌ی −180° + i × 22.5° (۱۶ پرتو، ۳۶۰ درجه)
- بدون بازتاب: ۰٫۰ یا بی‌نهایت؛ گردوغبار: ۰٫۰۵ متر، کمتر از حداقل برد
- پرتوهای غیرقابل استفاده NaN می‌شوند، نه صفر
- اسکن معتبر اگر دست‌کم ۵۰٪ پرتوها قابل استفاده باشند (مه ۴ اسکن از ۵۰ را رد می‌کند)
- nearest_obstacle(): تابع np.nanargmin روی آخرین اسکن معتبر
**Visual:** IMAGE ../assets/week02/fig_lidar_scan.png
**Notes:** چرا NaN و نه صفر؟ برد صفر به‌معنای مانعی چسبیده به ربات خوانده می‌شود. تمرین ۱، وظایف ۴ تا ۶.

## SLIDE 18 | مثال ۶: تبدیل GPS به متر محلی با دست
**Content:**
- یک درجه عرض = Rπ/180 = 111 195 m (R = 6 371 km)
- یک درجه طول = 111 195 × cos φ0؛ در مهرآباد (35.69°): 90 314 m
- 0.001° شمال و شرق مبدأ ← (شرق، شمال) = (90.3, 111.2) m
- رد کنید: HDOP > 5، مقدار NaN، |lat| > 90°
**Visual:** IMAGE ../assets/week02/fig_gps_enu.png
**Notes:** تقریب زمین تخت برگرفته از Groves, Ch. 2؛ خطا در ۸۰ متر کمتر از ۱ سانتی‌متر است. فراموش‌کردن cos φ0 یکی از جهش‌های (mutant) نمره‌دهی است.

## SLIDE 19 | آزمون کلاس‌ها با pytest
**Content:**
- فایل‌های test_*.py، توابع test_* و assert ساده
- pytest.raises: خطا باید رخ دهد
- @pytest.mark.parametrize: یک آزمون، ورودی‌های متعدد
- @pytest.fixture: یک شیء تازه برای هر آزمون
- آزمون خوب عدد را بررسی می‌کند، نه فقط نوع را
**Visual:** کد: fixture با نام `imu()`، آزمون `test_imu_removes_bias` و آزمون پارامتری `test_rate_must_be_positive`؛ خط ترمینال «41 passed».
**Notes:** برگرفته از GeeksforGeeks «Getting Started with Pytest». در تمرین ۱، آزمون‌های دانشجو روی سه پیاده‌سازی عمداً معیوب اجرا و نمره‌دهی می‌شوند.

## SLIDE 20 | کارهای این هفته
**Content:**
- آزمایشگاه: notebooks/week02/week02_oop_sensors.ipynb (۵ تمرین TODO)
- سر کلاس: ۴ مسئله‌ی گرم‌کردن (لیست مشترک، زنجیره‌ی super، IMU با دست، نرخ صعود)
- تمرین ۱: حسگرهای LiDAR و GPS به‌همراه آزمون‌های pytest؛ حدود ۴ ساعت؛ مهلت هفته‌ی ۳
- پیش از push: python -m pytest assignments/assignment01
**Visual:** سه نماد: نوت‌بوک، مداد، شاخه‌ی گیت.
**Notes:** آزمون‌های پنهان تمرین ۱ اعداد حاصل از فایل‌های CSV را بررسی می‌کنند؛ جدول خروجی مورد انتظار در نوت‌بوک امکان خودسنجی می‌دهد.

## SLIDE 21 | آنچه اکنون بلدیم انجام دهیم
**Content:**
- داده و رفتار را در کلاس جمع کنیم و از __init__ و self درست استفاده کنیم
- وضعیت را با property محافظت کنیم و وضعیت تغییرپذیر را برای هر شیء جدا نگه داریم
- با وراثت و super() کد را بازاستفاده کنیم و با ABC رابط را اجباری کنیم
- رکورد را با dataclass منجمد بنویسیم و حسگرها را چندریختی پردازش کنیم
- سلسله‌مراتب کلاس‌ها را با pytest بیازماییم
**Visual:** ربات اسلاید آغازین که هر سه حسگرش اکنون به یک جعبه‌ی «رابط BaseSensor» وصل است، با تیک سبز روی هر کدام.
**Notes:** به پرسش آغازین برگردید: حسگر چهارم یعنی یک زیرکلاس جدید با دو متد.
