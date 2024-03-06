import datetime
import pandas as pd
import seaborn as sn
import locale
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt



locale.setlocale(locale.LC_ALL,'en_US.UTF-8')

def getMostSoldItems(product_df: pd.DataFrame, order_items_df:pd.DataFrame):
    df = pd.merge(product_df, order_items_df, how='inner', on='product_id')
    return df.groupby(by=['product_category_name']).product_category_name.count().sort_values().head(10)

def getTotalOrder(order_df:pd.DataFrame, deliveredOnly:bool):
    return len(order_df if deliveredOnly == False else order_df[order_df.order_status == 'delivered'])

def getTotalIncome(order_df:pd.DataFrame, order_item_df:pd.DataFrame, deliveredOnly:bool):
    order_df = order_df[order_df.order_status == 'delivered'] if deliveredOnly == True else order_df
    #asumsi saya freight cost dibayar oleh pembeli jadi tidak masuk ke pendapatan
    df = pd.merge(order_df, order_item_df, how='inner',on='order_id')
    return locale.currency(df.price.sum(), grouping=True)

#rata rata barang terjual per hari
def getAverageSoldItems(order_df:pd.DataFrame):
    return order_df.groupby(by=['order_purchase_timestamp']).order_purchase_timestamp.value_counts().mean()

def getProductPaymentDistribute(product_df:pd.DataFrame, order_items_df:pd.DataFrame, order_payments_df:pd.DataFrame):
    df = pd.merge(product_df, order_items_df, how='inner', on='product_id')
    df = pd.merge(df, order_payments_df, how='inner',on='order_id')
    return df.groupby(by=['product_category_name', 'payment_type']).payment_type.count()

#korelasi antara berat produk dan harga produk(rata-rata per berat prosduknya)
def getCorrelatProduct(product_df:pd.DataFrame, order_items_df:pd.DataFrame):
    df = pd.merge(product_df, order_items_df, how='inner', on='product_id')
    #kategorikan berdasarkan berat produk dengan rata rata harga yang didapat (top 10 barang terberat)
    return df.groupby(by=['product_weight_g']).agg({'price':'mean'}).sort_values(by=['product_weight_g'],ascending=False).head(10).to_dict()

def createRFM(order_df:pd.DataFrame, order_items_df:pd.DataFrame):
    df = pd.merge(order_df, order_items_df, how='inner',on='order_id')
    df = df.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (last_purchase - x).max().days,
    'order_id': 'count',
    'price': 'sum'
    }).reset_index()
    df.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    return df

#cleaning data
def clean_data(df: pd.DataFrame):
    return df.drop_duplicates()

customers = clean_data(pd.read_csv("customers_dataset.csv"))
order_items = clean_data(pd.read_csv("order_items_dataset.csv"))
# order_payments = clean_data(pd.read_csv("order_payments_dataset.csv"))
orders = clean_data(pd.read_csv("orders_dataset.csv"))
product_translation = clean_data(pd.read_csv("product_category_name_translation.csv"))
products = clean_data(pd.read_csv("products_dataset.csv"))

products = pd.merge(products, product_translation, how='inner', on='product_category_name')
products.drop(columns="product_category_name", inplace=True)
products.rename(columns={"product_category_name_english":"product_category_name"}, inplace=True)
for i in orders.columns.tolist()[3:]:
    orders[i] = pd.to_datetime(orders[i])

# print(orders.groupby(by=['customer_id']).count()>=1)
# print(createRFM(orders, order_items))


print(customers[["customer_city","customer_state"]])