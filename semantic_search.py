from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


docs = [
    "Avto kredit olish shartlari",
    "Mashina krediti uchun hujjatlar",
    "Uy krediti foiz stavkasi",
    "Bank hisob ochish",
    "Ob havo bugun juda issiq",
    "Toshkentda yomg'ir yog'adi",
    "Valyuta kursi bugun"
]

query = "Avto kredit"

doc_emb = model.encode(docs)
query_emb = model.encode(query)




