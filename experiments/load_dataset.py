from sklearn.datasets import fetch_20newsgroups


def load_data():

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = dataset.data
    labels = dataset.target
    label_names = dataset.target_names

    return documents, labels, label_names


if __name__ == "__main__":

    docs, labels, names = load_data()

    print("Number of documents:", len(docs))
    print("\nExample document:\n")
    print(docs[0][:500])