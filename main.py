import math
import pandas as pd
import seaborn as sn
import locale
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def clean_data(df: pd.DataFrame):
    return df.drop_duplicates()


customers = clean_data(pd.read_csv("customers_dataset.csv"))
sellers = clean_data(pd.read_csv("sellers_dataset.csv"))
order_items = clean_data(pd.read_csv("order_items_dataset.csv"))
order_payments = clean_data(pd.read_csv("order_payments_dataset.csv"))
orders = clean_data(pd.read_csv("orders_dataset.csv"))
product_translation = clean_data(pd.read_csv("product_category_name_translation.csv"))
products = clean_data(pd.read_csv("products_dataset.csv"))


def Pearson_correlation(X, Y):
    if len(X) == len(Y):
        Sum_xy = sum((X - X.mean()) * (Y - Y.mean()))
        Sum_x_squared = sum((X - X.mean()) ** 2)
        Sum_y_squared = sum((Y - Y.mean()) ** 2)
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr


def getMostSoldItems(product_df: pd.DataFrame, order_items_df: pd.DataFrame):
    df = pd.merge(product_df, order_items_df, how="inner", on="product_id")
    return (
        df.groupby(by=["product_category_name"])
        .product_category_name.count()
        .sort_values(ascending=False)
        .head(10)
    )


def getTotalOrder(order_df: pd.DataFrame, deliveredOnly: bool):
    return len(
        order_df
        if deliveredOnly == False
        else order_df[order_df.order_status == "delivered"]
    )


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


def getAverageSoldItems(order_df: pd.DataFrame):
    return (
        order_df.groupby(by=["order_purchase_timestamp"])
        .order_purchase_timestamp.value_counts()
        .mean()
    )


def getProductPaymentDistribute(
    product_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    order_payments_df: pd.DataFrame,
):
    df = pd.merge(product_df, order_items_df, how="inner", on="product_id")
    df = pd.merge(df, order_payments_df, how="inner", on="order_id")
    return df.groupby(by=["product_category_name", "payment_type"]).payment_type.count()


def getCorrelatProduct(product_df: pd.DataFrame, order_items_df: pd.DataFrame):
    df = pd.merge(product_df, order_items_df, how="inner", on="product_id")
    # kategorikan berdasarkan berat produk dengan rata rata harga yang didapat (top 10 barang terberat)
    return (
        df.groupby(by=["product_weight_g"])
        .agg({"price": "mean"})
        .sort_values(by=["product_weight_g"], ascending=False)
        .reset_index()
        .head(500)
        .to_dict()
    )


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
    return (
        df.groupby(by=["seller_state"])
        .seller_state.count()
        .sort_values(ascending=False)
    )


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
        value= 0 if str(val) == 'nan' else round(val, 1),
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
    st.write("Produk dengan Penjualan Terbanyak")

    fig, ax = plt.subplots()
    mostSoldItem = getMostSoldItems(
        product_df=products, order_items_df=order_items
    ).to_dict()
    plt.barh(y=list(mostSoldItem.keys()), width=list(mostSoldItem.values()))
    plt.xlabel("Total Penjualan")
    st.pyplot(fig)

st.write("Korelasi Berat Produk dengan Harga Produk")
fig, ax = plt.subplots()
data = getCorrelatProduct(products, order_items)
sn.scatterplot(x=data["product_weight_g"].values(), y=data["price"].values())

print(
    Pearson_correlation(
        np.array(list(data["product_weight_g"].values())),
        np.array(list(data["price"].values())),
    )
)
plt.xlabel("Berat Produk (gram)")
plt.ylabel("Harga (USD)")
st.pyplot(fig)

###########################################################################

st.subheader("Analisa Metode Pembayaran")

with st.container():

    payDistribute = getProductPaymentDistribute(
        products, order_items, order_payments
    ).to_dict()
    method_payment = ["boleto", "credit_card", "debit_card", "voucher"]

    temp = {}
    for k, v in payDistribute.items():
        if k[0] in mostSoldItem.keys():
            if k[0] not in temp.keys():
                temp[k[0]] = {}
            temp[k[0]][k[1]] = v

    # fill another method with 0
    for k, v in temp.items():
        for i in method_payment:
            if i not in v.keys():
                temp[k][i] = 0

    barWidth = 0.25
    fig, ax = plt.subplots()

    boleto = list(map(lambda x: x["boleto"], temp.values()))
    credit_card = list(map(lambda x: x["credit_card"], temp.values()))
    debit_card = list(map(lambda x: x["debit_card"], temp.values()))
    voucher = list(map(lambda x: x["voucher"], temp.values()))

    br1 = np.arange(len(boleto))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    plt.barh(br1, boleto, color="r", height=barWidth, edgecolor="grey", label="Boleto")
    plt.barh(
        br2,
        credit_card,
        color="g",
        height=barWidth,
        edgecolor="grey",
        label="Credit Card",
    )
    plt.barh(
        br3,
        debit_card,
        color="b",
        height=barWidth,
        edgecolor="grey",
        label="Debit Card",
    )
    plt.barh(
        br4,
        voucher,
        color="hotpink",
        height=barWidth,
        edgecolor="grey",
        label="Voucher",
    )

    plt.ylabel("Product Name")
    plt.xlabel("Banyak Pemakaian")
    plt.yticks([r + barWidth for r in range(len(boleto))], mostSoldItem.keys())
    plt.legend()

    st.pyplot(fig)

###########################################################################

st.subheader("Geolocate Analysis")

data = getMostSellestCountries(orders, order_items, sellers).to_dict()
list_state = [f"{k} ({v} Pembelian)" for k, v in data.items()]

st.write("Negara Bagian dengan Penjualan Produk Terbanyak")
data = (
    getCorrelatBuyerSellerLocation(orders, order_items, customers, sellers)
    .groupby(by=["seller_state"])
    .value_counts()
    .to_dict()
)
temp = {}
for k, v in data.items():
    if k[0] not in temp.keys():
        temp[k[0]] = {}
    temp[k[0]][k[1]] = v

seller_state_option = st.selectbox(label="Negara Penjual", options=list_state)

if seller_state_option != None:
    with st.container():
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.bar(
            x=temp[seller_state_option.split(" ")[0]].keys(),
            height=temp[seller_state_option.split(" ")[0]].values(),
        )
        ax.set_xlabel("Negara Pembeli")
        ax.set_ylabel("Total Pembelian")
        ax.set_title("Distribusi Negara Penjual dengan Negara Pembeli")
        plt.xticks(rotation=60)
        st.pyplot(fig)

###########################################################################

st.subheader("RFM Analysis")

rfm_df = createRFM(orders, order_items)

col = st.columns(3, gap="large")

with col[0]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    avg_recency = round(rfm_df.recency.mean(), 2)
    st.metric("Avg Ketepatan Waktu Pembelian", value=avg_recency)
    data = rfm_df.sort_values(by="recency").reset_index().head(10)
    plt.bar(x=data.customer_id, height=data.recency)
    ax.set_xlabel("Customer ID")
    ax.set_ylabel("Recency")
    ax.set_title("Recency Distribution")
    st.pyplot(fig)

with col[1]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Avg Frekuensi Pembelian", value=avg_frequency)
    data = rfm_df.sort_values(by="frequency", ascending=False).reset_index().head(10)
    plt.bar(x=data.customer_id, height=data.frequency)
    ax.set_xlabel("Customer ID")
    ax.set_ylabel("Frequency")
    ax.set_title("Frequency Distribution")
    st.pyplot(fig)

with col[2]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    avg_moneter = locale.currency(round(rfm_df.monetary.mean()), grouping=True)
    st.metric("Avg Moneter Pembelian", value=avg_moneter)
    data = rfm_df.sort_values(by="monetary", ascending=False).reset_index().head(10)
    plt.bar(x=data.customer_id, height=data.monetary)
    ax.set_xlabel("Customer ID")
    ax.set_ylabel("Monetary")
    ax.set_title("Monetary Distribution")
    st.pyplot(fig)

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
st.write(rfm_df[["customer_id", "RFM_Score"]].head(10).transpose())
