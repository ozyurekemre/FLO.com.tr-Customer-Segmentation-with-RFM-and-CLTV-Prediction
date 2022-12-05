# Değişkenler
#master_id : Eşsiz müşteri numarası
#order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
#last_order_channel : En son alışverişin yapıldığı kanal
#first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
#last_order_date : Müşterinin yaptığı son alışveriş tarihi
#last_order_date_online : Müşterinin online platformda yaptığı son alışveriş tarihi
#last_order_date_offline : Müşterinin offline platformda yaptığı son alışveriş tarihi
#order_num_total_ever_online :Müşterinin online platformda yaptığı toplam alışveriş sayısı
#order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
#customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
#customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
#interested_in_categories_12 :  Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################
#Görev 1: Veriyi Anlama ve Hazırlama
###############################

#Adım 1: flo_data_20K.csv verisini okuyunuz.

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ =pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()
#Adım2 alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df.describe().T
df.head()
df.isnull().sum()

#Adım3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
#"customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.

df.dropna(inplace=True)


replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

#alternatif
columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

#Adım4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["TotalOrder"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

#Adım5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

date_veriable_list = ["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]

for veriable in date_veriable_list:
    df[veriable] = pd.to_datetime(df[veriable])
#################################################
#Görev 2: CLTV Veri Yapısının Oluşturulması
#################################################
#Adım1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

today_date = df["last_order_date"].max() + dt.timedelta(days=2)

#Adım2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
#Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["TotalOrder"]
cltv_df["monetary_cltv_avg"] = df["TotalPrice"] / df["TotalOrder"]

###############################################################################
#Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
###############################################################################
#Adım1: BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])


#3 ay

cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])
#6 ay

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])

plot_period_transactions(bgf)
plt.show(block=True)

#Adım2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
#dataframe'ine ekleyiniz.


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df["monetary_cltv_avg"])

#cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

###cltv_df.drop("expected_average_profit", 1)


#Adım3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_df.sort_values(by="cltv", ascending=False).head(20)

############################################################
#Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
############################################################

#Adım1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])


cltv_df.sort_values(by="cltv", ascending=False).head(50)

#Adım2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df.groupby("segment").agg(
    {"count", "mean", "sum"})

#Yapılan işlemler sonucu ana fokusumuzun a grubu olması gerektiği aşikardır, b ve c grubu birbirine yakın olduğu için birleşmeye gidilebilir ve d grubu göz ardı edilebilir.
















