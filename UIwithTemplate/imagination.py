## 이미지화 알고리즘을 삽입 하자.

#### RP 함수 ####
def toRPdata(tsdatas, dimension=1, time_delay=1, threshold=None, percentage=10, flatten=False):
    X = []
    rp = RecurrencePlot(dimension= dimension,
                        time_delay= time_delay,
                        threshold= threshold,
                        percentage= percentage,
                        flatten= flatten)
    for data in tsdatas:
        data_rp = rp.fit_transform(data)
        X.append(data_rp[0])

    return np.array(X)

#### GraminAngularField
def toGAFdata(tsdatas, image_size=1., sample_range=(-1, 1), method='summation', overlapping=False, flatten=False):
    X = []
    gaf = GramianAngularField(image_size=image_size,
                             sample_range=sample_range,
                             method=method,
                             overlapping=overlapping,
                             flatten=flatten)
    for data in tsdatas:
        data_gaf = gaf.fit_transform(data)
        X.append(data_gaf[0])
    return np.array(X)

#### MarkovTransitionField
def toMTFdata(tsdatas, image_size=1., n_bins=5, strategy='quantile', overlapping=False, flatten=False):
    X = []
    mtf = MarkovTransitionField(image_size=image_size,
                               n_bins=n_bins,
                               strategy=strategy,
                               overlapping=overlapping,
                               flatten=flatten)
    for data in tsdatas:
        data_mtf = mtf.fit_transform(data)
        X.append(data_mtf[0])
    return np.array(X)
