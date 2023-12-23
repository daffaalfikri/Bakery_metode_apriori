import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("Bakery.csv")

st.title("Market Analyst Bakery")


def get_data(Items='', DateTime=''):
    data = df.copy()
    filtered = data.loc[
        (data["Items"].str.contains(Items)) &
        (data["DateTime"].astype(str).str.contains(DateTime))
    ]
    return filtered if not filtered.empty else "No Result"


def user_input_features():
    TransactionNo = st.selectbox("TransactionNo", ['1', '2', '3', '4', '5'])
    Items = st.selectbox("Items", ['Bread', 'Scandinavian', 'Hot chocolate',
                                   'Jam ', 'Bread', 'Scandinavian', 'Hot chocolate', 'Scandinavian', 'Scandinavian'])
    DateTime = st.select_slider("DateTime", list(map(str, range(1, 42))))
    return TransactionNo, DateTime, Daypart


Items, DateTime, Daypart = user_input_features()

data = get_data(Items.lower(), year)


def get_apriori_results(transactions, min_support, min_confidence):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return frequent_itemsets, rules


def main():
    st.title("Apriori Algoritma Association")

    # Contoh data transaksi
    transactions = [
        ['Bread', 'Scandinavian', 'Hot chocolate'],
        ['Scandinavian', 'Hot chocolate'],
        ['Bread', 'Scandinavian', 'Bread'],
        ['Hot chocolate', 'Bread'],
        ['Scandinavian']
    ]

    # Parameter untuk algoritma Apriori
    min_support = st.slider("Minimal Support", 0.0, 0.8, 0.2)
    min_confidence = st.slider("Minimal Confidence", 0.0, 0.8, 0.7)

    # Mendapatkan hasil dari algoritma Apriori
    frequent_itemsets, rules = get_apriori_results(
        transactions, min_support, min_confidence)

    # Menampilkan hasil
    st.subheader("Frequent Itemsets:")
    st.write(frequent_itemsets)

    st.subheader("Association Rules:")
    st.write(rules)


if __name__ == "__main__":
    main()
