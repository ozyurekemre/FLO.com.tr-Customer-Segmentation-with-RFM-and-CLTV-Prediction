###############################################################
#RFM ile FLO Müşteri Segmentasyonu (FLO Customer Segmentation with RFM)
###############################################################
import datetime
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Adım1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
#Adım2: Veri setinde
#a. İlk 10 gözlem,
#b. Değişken isimleri,
#c. Betimsel istatistik,
#d. Boş değer,
#e. Değişken tipleri, incelemesi yapınız.


df.head()
df.tail()
df.shape
df.describe().T
df.index
df.columns
df.info()
df.isnull().values.any()
df.isnull().sum()
#Index(['master_id', 'order_channel', 'last_order_channel', 'first_order_date', 'last_order_date', 'last_order_date_online', 'last_order_date_offline', 'order_num_total_ever_online', 'order_num_total_ever_offline', 'customer_value_total_ever_offline', 'customer_value_total_ever_online', 'interested_in_categories_12', 'TotalPrice', 'TotalOrder'], dtype='object')


#Adım3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
#Her bir müşterinin toplamalışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["TotalOrder"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


#Adım4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df['first_order_date'].max()
df["last_order_date"].max()
df["last_order_date_online"].max()
df["last_order_date_offline"].max()



date_veriable_list = ["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]

for veriable in date_veriable_list:
    df[veriable] = pd.to_datetime(df[veriable])

Adım#5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.

df.groupby(["order_channel"]).agg({"master_id": "count","TotalOrder": "sum","TotalPrice": "sum"})


#Adım6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"TotalPrice": "sum"}).sort_values("TotalPrice", ascending=False).head(10)
#df.sort_values(by="TotalPrice", ascending=False).head(10)

#Adım7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"TotalOrder": "sum"}).sort_values("TotalOrder", ascending=False).head(10)
#df.sort_values(by="TotalOrder", ascending=False).head(10)

#Adım8: Veri ön hazırlık sürecini fonksiyonlaştırınız.

def rmf_(dataframe,date_veriable_list):
    dataframe["TotalOrder"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["TotalPrice"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    for veriable in date_veriable_list:
        dataframe[veriable] = pd.to_datetime(dataframe[veriable])
    return dataframe
###########################################
#Görev 2: RFM Metriklerinin Hesaplanması
###########################################
#Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

""" Recency = Mevcut gün ile müşterinin son alım yaptığı gün arasında geçen zaman """
""" Frequency = Müşterinin toplam sipariş sayısıdır """
""" Monetary = Müşterinin alışverişlerinden şirketin elde ettiği kazanç toplamı """

#Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
#Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.

#today_date = dt.datetime(2021,6,1)
#type(today_date)
today_date = df["last_order_date"].max() + dt.timedelta(days=2)

rfm = df.groupby('master_id').agg({'last_order_date': lambda lastorderdate: (today_date - lastorderdate.max()).days,
                                     'TotalOrder': lambda TotalOrder: TotalOrder,
                                     'TotalPrice': lambda TotalPrice: TotalPrice})

rfm.head()
#Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm.columns = ['recency', 'frequency', 'monetary']

######################################
#Görev 3: RF Skorunun Hesaplanması
#####################################
#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
#Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
#Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.


rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

######################################
#Görev 4: RF Skorunun Segment Olarak Tanımlanması
#####################################
#Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
#Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
######################################
#Görev 5: Aksiyon Zamanı !
#####################################
#Adım1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
#Adım2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.

#a###
women = df[df["interested_in_categories_12"] == '[KADIN]']

df_new = rfm[((rfm["segment"] == "champions")
       | (rfm["segment"] == "loyal_customers"))]

women_vip_cust = pd.merge(df_new,women[["interested_in_categories_12","master_id"]],on=["master_id"])
women_vip_cust.head()

women_vip_cust = women_vip_cust["master_id"]
women_vip_cust.to_csv("women_vip.cust.csv")

################## alternatif2 ????
target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["master_id"]

cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape
##################

#b###
b = rfm[(rfm["segment"] =="hibernating") |
        (rfm["segment"] == 'need_attention' ) |
        (rfm["segment"] == "new_customers")]

men_boy = df[(df["interested_in_categories_12"].str.contains('ERKEK') | df["interested_in_categories_12"].str.contains('COCUK'))]

men_boy_cust = pd.merge(b,men_boy[["interested_in_categories_12","master_id"]],on=["master_id"])
men_boy_cust.to_csv("men_boy_cust.cv")
###############alt2  ???
step_b = rfm[((rfm["segment"] == "cant_loose") |
             (rfm["segment"] == "hibernating") |
             (rfm["segment"] == "new_customers"))&
             ((rfm["interested_in_categories_12"].str.contains("ERKEK")) |
             (rfm["interested_in_categories_12"].str.contains("COCUK")))]
#####################