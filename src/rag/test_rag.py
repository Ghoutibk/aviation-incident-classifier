"""Test rapide du pipeline RAG sur 3 questions représentatives."""
from src.rag.chain import ask


QUESTIONS = [
    "Quels sont les risques liés au givrage en aviation légère ?",
    "Quels facteurs humains reviennent le plus souvent dans les accidents mortels ?",
    "Cite-moi des exemples d'incidents de maintenance sur hélicoptères.",
]


def main() -> None:
    for q in QUESTIONS:
        print("=" * 70)
        print(f"❓ {q}")
        print("=" * 70)

        response = ask(q, k=5)

        print(f"\n💡 RÉPONSE :\n{response.answer}\n")

        print("SOURCES:")
        for s in response.sources:
            print(f"  - {s.bea_reference} (similarité {s.similarity:.2f}) : {s.text[:100]}...")
        print()


if __name__ == "__main__":
    main()