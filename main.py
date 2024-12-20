import re
import pandas as pd
import locale
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from requests import get


locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


customers = load_data("data/customers_dataset.csv")
sellers = load_data("data/sellers_dataset.csv")
order_items = load_data("data/order_items_dataset.csv")
order_payments = load_data("data/order_payments_dataset.csv")
order_reviews = load_data("data/order_reviews_dataset.csv")
orders = load_data("data/orders_dataset.csv")
product_translation = load_data("data/product_category_name_translation.csv")
products = load_data("data/products_dataset.csv")


@st.cache_data
def Pearson_correlation(X, Y):
    if len(X) == len(Y):
        Sum_xy = sum((X - X.mean()) * (Y - Y.mean()))
        Sum_x_squared = sum((X - X.mean()) ** 2)
        Sum_y_squared = sum((Y - Y.mean()) ** 2)
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr


@st.cache_data
def decode_dict(data: dict):
    temp = {}
    for k, v in data.items():
        if k[0] not in temp.keys():
            temp[k[0]] = {}
        temp[k[0]][k[1]] = v
    return temp


@st.cache_data
def getMostSoldItems(product_df: pd.DataFrame, order_items_df: pd.DataFrame, n=5):
    df = pd.merge(product_df, order_items_df, how="inner", on="product_id")

    result = df["product_category_name"].value_counts()
    return (
        pd.DataFrame(
            data={"product_category_name": result.index, "total_sold": result.values}
        )
        .sort_values(by="total_sold", ascending=False)
        .head(n)
    )


@st.cache_data
def getTotalOrder(order_df: pd.DataFrame, deliveredOnly: bool):
    return len(
        order_df
        if deliveredOnly == False
        else order_df[order_df.order_status == "delivered"]
    )


@st.cache_data
def getTotalIncome(
    order_df: pd.DataFrame, order_item_df: pd.DataFrame, deliveredOnly: bool
):
    order_df = (
        order_df[order_df.order_status == "delivered"]
        if deliveredOnly == True
        else order_df
    )
    # asumsi saya freight cost dibayar oleh pembeli jadi tidak masuk ke pendapatan
    df = pd.merge(order_df, order_item_df, how="inner", on="order_id")
    return locale.currency(df.price.sum(), grouping=True)


@st.cache_data
def getAverageSoldItems(order_df: pd.DataFrame):
    return (
        order_df.groupby(by=["order_purchase_timestamp"])
        .order_purchase_timestamp.value_counts()
        .mean()
    )


@st.cache_data
def getProductPaymentDistribute(
    product_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    order_payments_df: pd.DataFrame,
    only_most_sold_item: bool = False,
    most_sold_item: pd.DataFrame = None,
):
    df = pd.merge(product_df, order_items_df, how="inner", on="product_id")
    df = pd.merge(
        df[["product_category_name", "order_id"]],
        order_payments_df[["order_id", "payment_type"]],
        how="inner",
        on="order_id",
    )

    if only_most_sold_item:
        df = df[df.product_category_name.isin(most_sold_item.product_category_name)]

    df = df[["product_category_name", "payment_type"]].value_counts(sort=False)

    result = pd.DataFrame(
        data={
            "product_category_name": df.index.map(lambda x: x[0]),
            "payment_type": df.index.map(lambda x: x[1]),
            "total_used": df.values,
        }
    )

    return result


@st.cache_data
def getCorrelatProduct(product_df: pd.DataFrame, order_items_df: pd.DataFrame):
    df = pd.merge(product_df, order_items_df, how="inner", on="product_id")
    # kategorikan berdasarkan berat produk dengan rata rata harga yang didapat (top 10 barang terberat)
    return (
        df.groupby(by=["product_weight_g"])
        .agg({"price": "mean"})
        .sort_values(by=["product_weight_g"], ascending=False)
        .reset_index()
        .to_dict()
    )


@st.cache_data
def createRFM(order_df: pd.DataFrame, order_items_df: pd.DataFrame):
    df = pd.merge(order_df, order_items_df, how="inner", on="order_id")
    last_purchase = order_df.order_purchase_timestamp.max()
    df = (
        df.groupby("customer_id")
        .agg(
            {
                "order_purchase_timestamp": lambda x: (last_purchase - x).max().days,
                "order_id": "count",
                "price": "sum",
            }
        )
        .reset_index()
    )
    df.columns = ["customer_id", "recency", "frequency", "monetary"]
    df.customer_id = df.customer_id.apply(lambda x: x[:5])
    return df


@st.cache_data
def getMostSellestCountries(
    order_df: pd.DataFrame, order_items_df: pd.DataFrame, seller_df: pd.DataFrame
):
    df = pd.merge(
        order_df[order_df.order_status != "canceled"],
        order_items_df,
        how="inner",
        on="order_id",
    )
    df = pd.merge(
        df,
        seller_df,
        how="inner",
        on="seller_id",
    )
    result = (
        df.groupby(by=["seller_state"])
        .seller_state.count()
        .sort_values(ascending=False)
    )

    return pd.DataFrame(
        data={"seller_state": result.keys(), "total_sold": result.values}
    )


@st.cache_data
def getCorrelatBuyerSellerLocation(
    order_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    seller_df: pd.DataFrame,
):
    df = pd.merge(
        order_df,
        customer_df[["customer_state", "customer_id"]],
        how="inner",
        on="customer_id",
    )
    df = pd.merge(
        df, order_items_df[["order_id", "seller_id"]], how="inner", on="order_id"
    )
    df = pd.merge(
        df, seller_df[["seller_id", "seller_state"]], how="inner", on="seller_id"
    )

    return df[["customer_state", "seller_state"]]


@st.cache_data
def getProductReview(
    products_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    order_reviews_df: pd.DataFrame,
):
    df = pd.merge(order_items_df, order_reviews_df, how="inner", on="order_id")
    df = pd.merge(
        df,
        products_df[["product_id", "product_category_name"]],
        how="inner",
        on="product_id",
    )

    return df[["product_id", "review_score"]].groupby(by=["product_id"])


@st.cache_data
def getCorrelatProductDescWithReview(
    products_df: pd.DataFrame,
    order_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    order_reviews_df: pd.DataFrame,
):
    df = pd.merge(
        order_df[order_df.order_status == "delivered"],
        order_items_df[["order_id", "product_id", "price"]],
        how="inner",
        on="order_id",
    )
    df = pd.merge(
        df,
        products_df[["product_id", "product_description_lenght"]],
        how="inner",
        on="product_id",
    )
    df = pd.merge(
        df, order_reviews_df[["order_id", "review_score"]], how="inner", on="order_id"
    )

    return (
        df[["product_id", "product_description_lenght", "review_score"]]
        .groupby(by=["product_id"])
        .agg({"review_score": "mean", "product_description_lenght": "mean"})
        .sort_values(by=["product_description_lenght"], ascending=False)
        .reset_index()
    )


@st.cache_data
def getSoldProduct(order_df: pd.DataFrame, order_items_df: pd.DataFrame):
    df = pd.merge(
        order_df[order_df.order_status == "delivered"],
        order_items_df,
        how="inner",
        on="order_id",
    )
    return (
        df.groupby(by=["product_id"])
        .order_id.count()
        .rename({"order_id": "total_penjualan"})
    )


@st.cache_data
def load_state_abbreviation():
    state = pd.read_html(get("https://brazil-help.com/brazilian_states.htm").content)[2]
    state.columns = state.iloc[1]
    state = state.iloc[2:]
    state = state[["Common Two Letter Abbreviation", "State"]]

    return dict(state.values)


#################################3#################################3#################################3

products = pd.merge(
    products, product_translation, how="inner", on="product_category_name"
)
products.drop(columns="product_category_name", inplace=True)
products.rename(
    columns={"product_category_name_english": "product_category_name"}, inplace=True
)

for i in orders.columns.tolist()[3:]:
    orders[i] = pd.to_datetime(orders[i])

first_date_order = orders.order_purchase_timestamp.min()
last_date_order = orders.order_purchase_timestamp.max()

with st.sidebar:
    first_date, last_date = st.date_input(
        label="Plese select date range",
        value=[first_date_order, last_date_order],
        max_value=last_date_order,
        min_value=first_date_order,
    )
    search = st.text_input("Check Order ID")
    with st.expander("Result: "):
        st.write(orders.loc[search == orders.order_id])

filtered_orders = orders[
    (orders["order_purchase_timestamp"] >= str(first_date))
    & (orders["order_purchase_timestamp"] <= str(last_date))
]
filtered_orders_items = pd.merge(
    order_items, filtered_orders, how="inner", on="order_id"
)

###########################################################################

st.header("E-Commerce Report")

###########################################################################

st.subheader("Analisa Penjualan")
col = st.columns([3, 3, 2], gap="medium")

with col[0]:
    st.metric(label="Total Penjualan", value=getTotalOrder(filtered_orders, False))
    st.metric(
        label="Total Penjualan (Delivered Only)",
        value=getTotalOrder(filtered_orders, True),
    )

with col[1]:
    st.metric(
        label="Total Pendapatan",
        value=getTotalIncome(filtered_orders, order_items, False),
    )
    st.metric(
        label="Total Pendapatan (Delivered Only)",
        value=getTotalIncome(filtered_orders, order_items, True),
    )

with col[2]:
    val = getAverageSoldItems(filtered_orders)
    st.metric(
        label="Rata-Rata Barang Terjual Per-Hari",
        value=0 if str(val) == "nan" else round(val, 1),
    )

###########################################################################

st.subheader("Analisa Produk")
col = st.columns(2, gap="medium")
with col[0]:
    st.write("Top 10 Product E-Commerce Terbanyak")

    fig, ax = plt.subplots()
    product_frequent = (
        products.groupby(by=["product_category_name"])
        .product_category_name.count()
        .sort_values(ascending=False)
        .head(7)
        .to_dict()
    )
    product_frequent = dict(
        sorted(product_frequent.items(), key=lambda x: x[1], reverse=True)
    )
    plt.pie(
        x=product_frequent.values(),
        labels=product_frequent.keys(),
        autopct="%1.1f%%",
        explode=list(
            map(
                lambda x: 0.2 if x == max(product_frequent.values()) else 0,
                product_frequent.values(),
            )
        ),
    )

    st.pyplot(fig)

with col[1]:
    st.write("Kategori Produk dengan Penjualan Terbanyak")

    fig, ax = plt.subplots()
    mostSoldItem = getMostSoldItems(product_df=products, order_items_df=order_items)
    st.bar_chart(
        mostSoldItem,
        x="product_category_name",
        y="total_sold",
        x_label="Kategori Produk",
        y_label="Total Terjual",
        color="total_sold",
    )

###########################################################################

st.subheader("Analisa Metode Pembayaran")

with st.container():

    payDistribute = getProductPaymentDistribute(
        products, order_items, order_payments, True, mostSoldItem
    )
    st.bar_chart(
        payDistribute,
        x="product_category_name",
        y="total_used",
        color="payment_type",
        stack="normalize",
        horizontal=True,
        x_label="Total Digunakan",
        y_label="Kategori Produk",
    )

###########################################################################

st.subheader("Analisa Kualitas Produk")

chooseProductCategory = st.selectbox(
    label="Pilih kategori produk",
    options=dict.fromkeys(products.product_category_name.sort_values()),
)

scoreReview = [i for i in range(1, 6)]

st.code(f"Produk Best Seller dari Kategori {chooseProductCategory}")

data = getProductReview(
    products[products.product_category_name == chooseProductCategory],
    order_items,
    order_reviews,
)

qtySoldProduct = getSoldProduct(orders, order_items)

produkReview = decode_dict(data.value_counts().to_dict())

for k, v in produkReview.items():
    for j in scoreReview:
        if j not in v.keys():
            produkReview[k][j] = 0

data = pd.merge(
    data.agg({"review_score": "count"}),
    data.agg({"review_score": "mean"}),
    how="inner",
    on="product_id",
)

data.rename(
    columns={"review_score_x": "total_penilaian", "review_score_y": "review_score"},
    inplace=True,
)

data["review__score_list"] = produkReview.values()

data.sort_values(by=["total_penilaian", "review_score"], ascending=False, inplace=True)

col = st.columns(2)
for i in range(len(col)):
    with col[i]:
        product_id = data.index.values[i]
        st.write(f"Produk ID :{product_id}")
        st.write(
            f"{round(data.at[product_id,'review_score'],1)}:star: ({data.at[product_id,'total_penilaian']} Ulasan, {qtySoldProduct.loc[product_id]} Penjualan)"
        )

        inner_col = st.columns(5)
        for j in range(len(inner_col)):
            with inner_col[j]:
                score = data.at[product_id, "review__score_list"][j + 1]

                st.button(
                    label=f"{j+1}:star: ({score})",
                    disabled=True,
                    key=np.random.random(),
                )

###########################################################################

st.subheader("Geolocate Analysis")

data = getMostSellestCountries(orders, order_items, sellers)
state_abbreviation = load_state_abbreviation()

data.replace(state_abbreviation, inplace=True)

list_state = [f"{k} ({v} Pembelian)" for k, v in data.to_numpy()]

st.write("Negara Bagian dengan Penjualan Produk Terbanyak")

data = (
    getCorrelatBuyerSellerLocation(orders, order_items, customers, sellers)
    .groupby(by=["seller_state"])
    .value_counts()
    .reset_index(level=["seller_state", "customer_state"])
)

data.replace(state_abbreviation, inplace=True)

seller_state_option = st.selectbox(label="Negara Penjual", options=list_state)

if seller_state_option != None:
    with st.container():
        chosen_state = re.match(r"^(.+?) \(", seller_state_option).group(1)

        st.bar_chart(
            data[data.seller_state == chosen_state],
            x="customer_state",
            y="count",
            x_label="Negara Pembeli",
            y_label="Total Pembelian",
            color="count",
        )

###########################################################################

st.subheader("RFM Analysis")

rfm_df = createRFM(orders, order_items)

col = st.columns(3, gap="large")

with col[0]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    avg_recency = round(rfm_df.recency.mean(), 2)
    data = rfm_df.sort_values(by="recency").reset_index().head(10)

    st.metric("Avg Ketepatan Waktu Pembelian", value=avg_recency)
    st.bar_chart(
        data,
        x="customer_id",
        y="recency",
        x_label="Customer ID",
        y_label="Recency",
        height=200,
    )

with col[1]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    data = rfm_df.sort_values(by="frequency", ascending=False).reset_index().head(10)

    st.metric("Avg Frekuensi Pembelian", value=avg_frequency)
    st.bar_chart(
        data,
        x="customer_id",
        y="frequency",
        x_label="Customer ID",
        y_label="Frequency",
        height=200,
    )

with col[2]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    avg_moneter = locale.currency(round(rfm_df.monetary.mean()), grouping=True)
    data = rfm_df.sort_values(by="monetary", ascending=False).reset_index().head(10)

    st.metric("Avg Moneter Pembelian", value=avg_moneter)
    st.bar_chart(
        data,
        x="customer_id",
        y="monetary",
        x_label="Customer ID",
        y_label="Monetary",
        height=200,
    )

# formula from gfg source
rfm_df["R_rank"] = rfm_df.recency.rank(ascending=False)
rfm_df["F_rank"] = rfm_df.frequency.rank(ascending=True)
rfm_df["M_rank"] = rfm_df.monetary.rank(ascending=True)

rfm_df["R_rank_norm"] = (rfm_df["R_rank"] / rfm_df["R_rank"].max()) * 100
rfm_df["F_rank_norm"] = (rfm_df["F_rank"] / rfm_df["F_rank"].max()) * 100
rfm_df["M_rank_norm"] = (rfm_df["F_rank"] / rfm_df["M_rank"].max()) * 100

rfm_df.drop(columns=["R_rank", "F_rank", "M_rank"], inplace=True)

rfm_df["RFM_Score"] = (
    0.15 * rfm_df["R_rank_norm"]
    + 0.28 * rfm_df["F_rank_norm"]
    + 0.57 * rfm_df["M_rank_norm"]
)
rfm_df["RFM_Score"] *= 0.05
rfm_df = rfm_df.round(2).sort_values(by="RFM_Score", ascending=False).reset_index()

st.write("Top 10 Highest RFM Score")
st.write(
    rfm_df[["customer_id", "RFM_Score"]]
    .head(10)
    .apply({"customer_id": str, "RFM_Score": str})
    .transpose()
)
