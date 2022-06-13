
def intersection_listcomprehension(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def intersection(a_subset, a_set):
    intersect = []
    for c in a_set:
        if c in a_subset:
            intersect.append(c)
    return intersect

def remove_nacolumns(df):
    df = df.dropna(axis=1, how='all')
    return df

def remove_alpha(df):
    df = df.replace("[^0-9.-]", '', regex=True)
    return df

