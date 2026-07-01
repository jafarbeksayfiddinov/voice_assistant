DOCS = [
    # 🚗 Car / Auto loans
    "Avto kredit olish shartlari",
    "Mashina krediti uchun hujjatlar",
    "Avtomobil uchun foiz stavkasi",
    "Car loan interest rate in bank",
    "Vehicle financing options",

    # 🏠 Home loans
    "Uy krediti olish tartibi",
    "Ipoteka kredit shartlari",
    "Mortgage loan requirements",
    "Uy sotib olish uchun kredit",

    # 🏦 Banking basics
    "Bank hisob ochish",
    "Debit kartani qanday olish mumkin",
    "Bank xizmatlari ro'yxati",
    "Online banking tizimi",

    # 💰 Credit / loans general
    "Kredit olish uchun talablar",
    "Microloan tez olish",
    "Credit history nima",
    "Loan approval process",

    # 💱 Finance / currency
    "Valyuta kursi bugun",
    "USD to UZS exchange rate",
    "Currency exchange in Uzbekistan",
    "Money transfer fees",

    # 🌤️ Unrelated (for testing semantic separation)
    "Ob havo bugun juda issiq",
    "Toshkentda yomg'ir yog'adi",
    "Bugun havo sovuq bo'ladi",
    "Weather forecast today",

    # 🍔 Random noise (VERY IMPORTANT for testing)
    "Football match results today",
    "How to cook pasta",
    "Best programming languages 2026"
]
PATH="info_list-2-merged.pdf"

QUESTIONS = [
    "Mashina krediti foizi qancha?",
    "Mahalla tadbirkori uchun kredit bormi?",
    "O'zini o'zi band qilganlar uchun kredit"
]

test_questions = [
    "Mashina krediti foizi qancha?",
    "UzAuto mashinasi uchun qarz kerak.",
    "Tadbirkorlik boshlash uchun mablag' olmoqchiman.",
    "O'zimni o'zim band qilganman, kredit bormi?",
    "50 million so'm qarz kerak.",
    "Qaysi kreditning foizi 0%?",
    "24 oyga mikroqarz olsam bo'ladimi?",
    "Har oy to'lovlari eng kam bo'lgan kredit qaysi?",
    "Avtomobil sotib olish uchun moliyalashtirish kerak.",
    "Biznesim uchun kichik kredit izlayapman."
]
test_questions_tiny = [
    "Mashina krediti foizi qancha?",
    "Tadbirkorlik uchun kredit kerak",
    "O'zimni o'zim band qilganman. Menga kredit kerak.",
]

test_ozband= "O'zimni o'zim band qilganman.",



DOC_SYSTEM = """
Sen SQB bankining virtual yordamchisisan.

VAZIFA:
Foydalanuvchi savoliga FAQAT berilgan KONTEXT asosida javob ber.

QOIDALAR:

1. FAQAT o'zbek tilida javob ber.
2. Javob 1 jumladan oshmasin.
3. Agar javob KONTEXT ichida aniq mavjud bo'lsa, faqat javobni qaytar.
4. Agar javob KONTEXT ichida mavjud bo'lmasa, faqat:
   Bilmayman.
5. Taxmin qilma.
6. O'zingdan ma'lumot qo'shma.
7. Javobdan oldin yoki keyin izoh yozma.
8. "Savol:" yoki "Javob:" so'zlarini yozma.
9. Faqat yakuniy javobni qaytar.
10. "Faqat javobni beram", "Mana javob", "Javob shu" kabi kirish so'zlarini ISHLATMA. Javobni to'g'ridan-to'g'ri raqam/fakt bilan boshla.

KONTEXT:
{context}
"""




OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "mistral:7b"