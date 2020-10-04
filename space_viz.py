#%%
import pandas as pd
import glob
import numpy as np
import math

#%%
def parse_single(year):
    PUS_start = pd.DataFrame()
    useful_cols = [
        "WAGP",
        "SEX",
        "AGEP",
        "DECADE",
        "RAC2P",
        "RAC1P",
        "SCHL",
        "WKW",
        "WKHP",
        "OCCP",
        "POWSP",
        "ST",
        "HISP",
    ]
    path = "data/data_raw/%s" % year
    PUS_start = pd.concat(
        [pd.read_csv(f, usecols=useful_cols) for f in glob.glob(path + "/*.csv")],
        ignore_index=True,
    )
    return PUS_start


def mapping_features(df):
    # entry date
    df["RACE"] = df["RAC2P"].map(
        lambda y: "White"
        if y == 1
        else "Black"
        if y == 2
        else "American Indian"
        if y <= 29
        else "Native Alaskan"
        if y <= 37
        else y
        if y <= 58
        else "Hispanic"
        if y == 70
        else np.nan
    )

    df["DECADE"] = df["DECADE"].replace(np.nan, 0)
    df["DECADE"] = df["DECADE"].map(
        lambda y: "Born in US"
        if y == 0
        else "Before 1950"
        if y == 1
        else "1950 - 1959"
        if y == 2
        else "1960 - 1969"
        if y == 3
        else "1970 - 1979"
        if y == 4
        else "1980 - 1989"
        if y == 5
        else "1990 - 1999"
        if y == 6
        else "2000 - 2009"
        if y == 7
        else "2010 or later"
        if y == 8
        else np.nan
    )

    # Race
    df["RAC2P"] = np.where(df["HISP"] == 1, df["RAC2P"], 70)
    df["RACE"] = df["RAC2P"].map(
        lambda y: "White"
        if y == 1
        else "Black"
        if y == 2
        else "American Indian"
        if y <= 29
        else "Native Alaskan"
        if y <= 37
        else y
        if y <= 58
        else "Hispanic"
        if y == 70
        else np.nan
    )

    df["RAC2P"] = np.where(df["HISP"] == 1, df["RAC2P"], 70)
    df["RACE2"] = df["RAC2P"].map(
        lambda y: "White"
        if y == 1
        else "Black"
        if y == 2
        else "American Indian"
        if y <= 29
        else "Native Alaskan"
        if y <= 37
        else "Asian"
        if y <= 58
        else "Hispanic"
        if y == 70
        else np.nan
    )

    # Sex
    df["SEX"] = df["SEX"].map(
        lambda y: "Male" if y == 1 else "Female" if y == 2 else "na"
    )

    # AGE
    df["AGE"] = df["AGEP"].map(
        lambda y: "0-17"
        if y <= 18
        else "18-24"
        if y <= 24
        else "25-54"
        if y <= 54
        else "55-64"
        if y <= 64
        else "65+"
    )

    # Education
    df["EDU"] = df["SCHL"].map(
        lambda y: "No_Highschool"
        if y <= 15
        else "Highschool"
        if y <= 17
        else "Some_College"
        if y <= 19
        else "Some_College"
        if y == 20
        else "B.S._Degree"
        if y == 21
        else "M.S._Degree"
        if y == 22
        else "PhD_or_Prof"
        if y <= 24
        else np.nan
    )

    # Occupation
    df["JOB"] = df["OCCP"].map(
        lambda y: "Business"
        if y <= 960
        else "Science"
        if y <= 1980
        else "Art"
        if y <= 2970
        else "Healthcare"
        if y <= 3550
        else "Services"
        if y <= 4655
        else "Sales"
        if y <= 5940
        else "Maintenance"
        if y <= 7640
        else "Production"
        if y <= 8990
        else "Transport"
        if y <= 9760
        else "Military"
        if y <= 9830
        else np.nan
    )

    return df


# %%
df_raw = parse_single(2018)

#%%
df_raw = parse_single(2018)
df = mapping_features(df_raw)
groupby_col = ["RACE", "DECADE"]
agg_df = df.groupby(groupby_col).count().reset_index()
agg_df = agg_df.iloc[:, 0 : len(groupby_col) + 1]

coding_df = pd.read_csv("data/data_raw/race_coding.csv")
result = pd.merge(agg_df, coding_df, how="left", on="RACE")
result["% Total Pop"] = result["ST"] / result["ST"].sum()

asian_result = result.dropna()
asian_result["% Asian Pop"] = asian_result["ST"] / asian_result["ST"].sum()

recomb_df = pd.DataFrame()

for race in asian_result["Asian"].unique():
    subset_df = asian_result[asian_result["Asian"] == race]
    subset_df = asian_result[asian_result["Asian"] == race]
    asian_result.iloc[:, 1] != "na"
    subset_df["% Race Pop"] = subset_df["ST"] / subset_df["ST"].sum()

    recomb_df = pd.concat([recomb_df, subset_df])


def panda_strip(x):
    r = []
    for y in x:
        if isinstance(y, str):
            y = y.strip()

        r.append(y)
    return pd.Series(r)


recomb_df = recomb_df.apply(lambda x: panda_strip(x))
recomb_df[["Asian"]] = recomb_df[["Asian"]].apply(lambda x: x.str.split().str[0])

wide_format = recomb_df.pivot(
    index="Asian", columns="DECADE", values="% Race Pop"
).reset_index()


wide_format = wide_format[
    [
        "Asian",
        "Born in US",
        "Before 1950",
        "1950 - 1959",
        "1960 - 1969",
        "1970 - 1979",
        "1980 - 1989",
        "1990 - 1999",
        "2000 - 2009",
        "2010 or later",
    ]
]

wide_format["% Immigrated"] = 1 - wide_format["Born in US"]

wide_format = wide_format.rename(columns={"Asian": "RACE2"})
wide_format = wide_format.replace(np.nan, 0)

wide_format.to_csv("data/data_output/imm_output.csv")


# %%
df_raw = parse_single(2018)
df = mapping_features(df_raw)
groupby_col = ["RACE2", "DECADE"]
agg_df = df.groupby(groupby_col).count().reset_index()
agg_df = agg_df.iloc[:, 0 : len(groupby_col) + 1]

recomb_df2 = pd.DataFrame()
for race in agg_df["RACE2"].unique():
    subset_df = agg_df[agg_df["RACE2"] == race]
    subset_df.iloc[:, 1] != "na"
    subset_df["% Race Pop"] = subset_df["ST"] / subset_df["ST"].sum()
    recomb_df2 = pd.concat([recomb_df2, subset_df])

wide_format2 = recomb_df2.pivot(
    index="RACE2", columns="DECADE", values="% Race Pop"
).reset_index()

wide_format2 = wide_format2[
    [
        "RACE2",
        "Born in US",
        "Before 1950",
        "1950 - 1959",
        "1960 - 1969",
        "1970 - 1979",
        "1980 - 1989",
        "1990 - 1999",
        "2000 - 2009",
        "2010 or later",
    ]
]

wide_format2["% Immigrated"] = 1 - wide_format2["Born in US"]

wide_format2.to_csv("data/data_output/imm_all_output.csv")

wide_format_comb = pd.concat([wide_format, wide_format2])
for col in wide_format_comb:
    if col != "RACE2":
        wide_format_comb[col] = wide_format_comb[col].astype(float).map("{:.2%}".format)
wide_format_comb.to_csv("data/data_output/imm_comb_output.csv")

recomb_df = recomb_df.rename(columns={"Asian": "RACE2"})
recomb_df["% Race Pop"] = recomb_df["% Race Pop"].astype(float).map("{:.2%}".format)
recomb_df2["% Race Pop"] = recomb_df2["% Race Pop"].astype(float).map("{:.2%}".format)
recomb_full = pd.concat([recomb_df, recomb_df2])

recomb_full.to_csv("data/data_output/imm_long.csv")
recomb_df.to_csv("data/data_output/imm_long1.csv")
recomb_df2.to_csv("data/data_output/imm_long2.csv")


#%%
df_raw = parse_single(2018)
df = mapping_features(df_raw)
groupby_col = ["RACE", "EDU"]
agg_df = df[df["AGE"] != "0-17"]
agg_df = agg_df.groupby(groupby_col).count().reset_index()
agg_df = agg_df.iloc[:, 0 : len(groupby_col) + 1]

coding_df = pd.read_csv("data/data_raw/race_coding.csv")
result = pd.merge(agg_df, coding_df, how="left", on="RACE")
result["% Total Pop"] = result["ST"] / result["ST"].sum()

asian_result = result.dropna()
asian_result["% Asian Pop"] = asian_result["ST"] / asian_result["ST"].sum()

recomb_df = pd.DataFrame()

for race in asian_result["Asian"].unique():
    subset_df = asian_result[asian_result["Asian"] == race]
    subset_df = asian_result[asian_result["Asian"] == race]
    asian_result.iloc[:, 1] != "na"
    subset_df["% Race Pop"] = subset_df["ST"] / subset_df["ST"].sum()

    recomb_df = pd.concat([recomb_df, subset_df])


def panda_strip(x):
    r = []
    for y in x:
        if isinstance(y, str):
            y = y.strip()

        r.append(y)
    return pd.Series(r)


recomb_df = recomb_df.apply(lambda x: panda_strip(x))
recomb_df[["Asian"]] = recomb_df[["Asian"]].apply(lambda x: x.str.split().str[0])

wide_format = recomb_df.pivot(
    index="Asian", columns="EDU", values="% Race Pop"
).reset_index()
wide_format.loc[wide_format["Asian"] == "", "Asian"] = (
    wide_format["Asian"].str.split().str.get(0)
)
wide_format = wide_format[
    [
        "Asian",
        "No_Highschool",
        "Highschool",
        "Some_College",
        "B.S._Degree",
        "M.S._Degree",
        "PhD_or_Prof",
    ]
]

wide_format["B.S. Degree +"] = (
    1
    - wide_format["No_Highschool"]
    - wide_format["Highschool"]
    - wide_format["Some_College"]
)

wide_format["% Completed Highschool"] = 1 - wide_format["No_Highschool"]

wide_format = wide_format.rename(columns={"Asian": "RACE2"})
wide_format.to_csv("data/data_output/edu_output.csv")

# %%
df_raw = parse_single(2018)
df = mapping_features(df_raw)
groupby_col = ["RACE2", "EDU"]
agg_df = df[df["AGE"] != "0-17"]
agg_df = agg_df.groupby(groupby_col).count().reset_index()
agg_df = agg_df.iloc[:, 0 : len(groupby_col) + 1]

recomb_df2 = pd.DataFrame()
for race in agg_df["RACE2"].unique():
    subset_df = agg_df[agg_df["RACE2"] == race]
    subset_df.iloc[:, 1] != "na"
    subset_df["% Race Pop"] = subset_df["ST"] / subset_df["ST"].sum()
    recomb_df2 = pd.concat([recomb_df2, subset_df])

wide_format2 = recomb_df2.pivot(
    index="RACE2", columns="EDU", values="% Race Pop"
).reset_index()

wide_format2 = wide_format2[
    [
        "RACE2",
        "No_Highschool",
        "Highschool",
        "Some_College",
        "B.S._Degree",
        "M.S._Degree",
        "PhD_or_Prof",
    ]
]

wide_format2["B.S. Degree +"] = (
    1
    - wide_format2["No_Highschool"]
    - wide_format2["Highschool"]
    - wide_format2["Some_College"]
)

wide_format2["% Completed Highschool"] = 1 - wide_format2["No_Highschool"]

wide_format2.to_csv("data/data_output/edu_all_output.csv")

wide_format_comb = pd.concat([wide_format, wide_format2])
for col in wide_format_comb:
    if col != "RACE2":
        wide_format_comb[col] = wide_format_comb[col].astype(float).map("{:.2%}".format)
wide_format_comb.to_csv("data/data_output/edu_comb_output.csv")

recomb_df = recomb_df.rename(columns={"Asian": "RACE2"})
recomb_df["% Race Pop"] = recomb_df["% Race Pop"].astype(float).map("{:.2%}".format)
recomb_df2["% Race Pop"] = recomb_df2["% Race Pop"].astype(float).map("{:.2%}".format)
recomb_full = pd.concat([recomb_df, recomb_df2])

recomb_full.to_csv("data/data_output/edu_long.csv")
recomb_df.to_csv("data/data_output/edu_long1.csv")
recomb_df2.to_csv("data/data_output/edu_long2.csv")
# %%
#####
# OCCUPATION
df_raw = parse_single(2018)
df = mapping_features(df_raw)
groupby_col = ["RACE", "JOB"]
agg_df = df[df["AGE"] != "0-17"]
agg_df = agg_df.groupby(groupby_col).count().reset_index()
agg_df = agg_df.iloc[:, 0 : len(groupby_col) + 1]

coding_df = pd.read_csv("data/data_raw/race_coding.csv")
result = pd.merge(agg_df, coding_df, how="left", on="RACE")
result["% Total Pop"] = result["ST"] / result["ST"].sum()

asian_result = result.dropna()
asian_result["% Asian Pop"] = asian_result["ST"] / asian_result["ST"].sum()

recomb_df = pd.DataFrame()

for race in asian_result["Asian"].unique():
    subset_df = asian_result[asian_result["Asian"] == race]
    subset_df = asian_result[asian_result["Asian"] == race]
    asian_result.iloc[:, 1] != "na"
    subset_df["% Race Pop"] = subset_df["ST"] / subset_df["ST"].sum()
    recomb_df = pd.concat([recomb_df, subset_df])

recomb_df = recomb_df.apply(lambda x: panda_strip(x))
recomb_df[["Asian"]] = recomb_df[["Asian"]].apply(lambda x: x.str.split().str[0])

wide_format = recomb_df.pivot(
    index="Asian", columns="JOB", values="% Race Pop"
).reset_index()
wide_format.loc[wide_format["Asian"] == "", "Asian"] = (
    wide_format["Asian"].str.split().str.get(0)
)

wide_format["% STEM"] = wide_format["Science"] + wide_format["Healthcare"]

wide_format = wide_format.rename(columns={"Asian": "RACE2"})
wide_format.to_csv("data/data_output/JOB_output.csv")

# %%
df_raw = parse_single(2018)
df = mapping_features(df_raw)
groupby_col = ["RACE2", "JOB"]
agg_df = df[df["AGE"] != "0-17"]
agg_df = agg_df.groupby(groupby_col).count().reset_index()
agg_df = agg_df.iloc[:, 0 : len(groupby_col) + 1]

recomb_df2 = pd.DataFrame()
for race in agg_df["RACE2"].unique():
    subset_df = agg_df[agg_df["RACE2"] == race]
    subset_df.iloc[:, 1] != "na"
    subset_df["% Race Pop"] = subset_df["ST"] / subset_df["ST"].sum()
    recomb_df2 = pd.concat([recomb_df2, subset_df])

wide_format2 = recomb_df2.pivot(
    index="RACE2", columns="JOB", values="% Race Pop"
).reset_index()


wide_format2["% STEM"] = wide_format2["Science"] + wide_format2["Healthcare"]

wide_format2.to_csv("data/data_output/JOB_all_output.csv")

wide_format_comb = pd.concat([wide_format, wide_format2])
for col in wide_format_comb:
    if col != "RACE2":
        wide_format_comb[col] = wide_format_comb[col].astype(float).map("{:.2%}".format)
wide_format_comb.to_csv("data/data_output/JOB_comb_output.csv")

recomb_df = recomb_df.rename(columns={"Asian": "RACE2"})
recomb_df["% Race Pop"] = recomb_df["% Race Pop"].astype(float).map("{:.2%}".format)
recomb_df2["% Race Pop"] = recomb_df2["% Race Pop"].astype(float).map("{:.2%}".format)
recomb_full = pd.concat([recomb_df, recomb_df2])

recomb_full.to_csv("data/data_output/job_long.csv")
recomb_df.to_csv("data/data_output/job_long1.csv")
recomb_df2.to_csv("data/data_output/job_long2.csv")

# %%
#####
# WAGE
def full_time_detect(df):
    df = df.loc[df.WKW < 4].copy()  # more than 40 weeks a year is considered full time
    df = df.loc[df.WKHP >= 35].copy()  # >=35 hr a week is considered full time
    df = df.loc[df.AGEP >= 18].copy()  # lower limit age
    df = df.loc[df.AGEP <= 70].copy()  # upper limit age
    return df


# determine who is considered an outlier
def outlier_wage(df):
    # wage_iqr = np.percentile(df.WAGP, 75) - np.percentile(df.WAGP, 25)
    # wage_upper = np.percentile(df.WAGP, 75) + wage_iqr * 3
    df = df.loc[df.WAGP >= 12500].copy()  # used because 12500 is poverty line
    df = df.loc[df.WAGP <= 400000].copy()  # used as ~1% wage US population
    return df


df_raw = parse_single(2018)
df = mapping_features(df_raw)
other = full_time_detect(df)
other = outlier_wage(other)
other = other[["RACE", "WAGP", "RACE2"]]
other.to_csv("data/data_output/wage_raw.csv")

#%%
agg_df = (
    other.groupby("RACE")
    .agg(count=("WAGP", "size"), mean=("WAGP", "mean"), median=("WAGP", "median"))
    .reset_index()
)


agg_df2 = (
    other.groupby("RACE2")
    .agg(count=("WAGP", "size"), mean=("WAGP", "mean"), median=("WAGP", "median"))
    .reset_index()
)
agg_df2["RACE"] = agg_df2["RACE2"]
agg_df3 = agg_df2[agg_df2["RACE"] != "Asian"]

coding_df = pd.read_csv("data/data_raw/race_coding.csv")
result = pd.merge(agg_df, coding_df, how="left", on="RACE")

result = result.apply(lambda x: panda_strip(x))
result[["Asian"]] = result[["Asian"]].apply(lambda x: x.str.split().str[0])

asian = result[0:21].rename(columns={"Asian": "RACE2"})
asian["RACE"] = "Asian"

# other_race = result[21:26].rename(columns={"RACE": "RACE2"})
wage_result = pd.concat([asian, agg_df2])
wage_result.to_csv("data/data_output/wage_asian_output.csv")

wage_result = pd.concat([asian, agg_df3])
wage_result.to_csv("data/data_output/wage_no_asian_output.csv")

#%%
df_raw = parse_single(2018)
df = mapping_features(df_raw)
other = full_time_detect(df)
other = outlier_wage(other)
other = other[["RACE", "WAGP", "RACE2"]]
other.to_csv("data/data_output/wage_raw.csv")

#%%
df_raw = parse_single(2018)
df = mapping_features(df_raw)
other = full_time_detect(df)
other = outlier_wage(other)
other = other[["RACE", "WAGP", "RACE2"]]
other.to_csv("data/data_output/wage_raw.csv")


agg_df = (
    other.groupby("RACE")
    .agg(count=("WAGP", "size"), mean=("WAGP", "mean"), median=("WAGP", "median"))
    .reset_index()
)


agg_df2 = (
    other.groupby("RACE2")
    .agg(count=("WAGP", "size"), mean=("WAGP", "mean"), median=("WAGP", "median"))
    .reset_index()
)
agg_df2["RACE"] = agg_df2["RACE2"]
agg_df3 = agg_df2[agg_df2["RACE"] != "Asian"]

coding_df = pd.read_csv("data/data_raw/race_coding.csv")
result = pd.merge(agg_df, coding_df, how="left", on="RACE")

result = result.apply(lambda x: panda_strip(x))
result[["Asian"]] = result[["Asian"]].apply(lambda x: x.str.split().str[0])

asian = result[0:21].rename(columns={"Asian": "RACE2"})
asian["RACE"] = "Asian"

# other_race = result[21:26].rename(columns={"RACE": "RACE2"})
wage_result = pd.concat([asian, agg_df2])
wage_result.to_csv("data/data_output/wage_asian_output.csv")

wage_result = pd.concat([asian, agg_df3])
wage_result.to_csv("data/data_output/wage_no_asian_output.csv")


#%%
long2 = pd.read_csv("data/data_output/edu_long2.csv")
long2 = long2.groupby("RACE2").sum().reset_index()

long2["% Pop"] = long2["ST"] / long2["ST"].sum()
long2["% Pop"] = long2["% Pop"].astype(float).map("{:.2%}".format)
long2["Race1"] = long2["RACE2"] + " (" + long2["% Pop"] + ")"
long2["Race2"] = long2["RACE2"] + " (" + long2["% Pop"] + ")"
long2 = long2[long2["RACE2"] != "Asian"]

long2
# %%
long1 = pd.read_csv("data/data_output/edu_long1.csv")
long1 = long1.groupby("RACE2").sum().reset_index()

long1["% Pop"] = long1["ST"] / long1["ST"].sum()
long1["% Pop"] = long1["% Pop"].astype(float).map("{:.2%}".format)
long1["Race1"] = "Asian (5.42%)"
long1["Race2"] = long1["RACE2"] + " (" + long1["% Pop"] + ")"
long1
# %%
long_combo = pd.concat([long2, long1])
long_combo.to_csv("data/data_output/long_combo.csv")

# %%
