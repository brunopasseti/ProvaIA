def correlation_matrix(df):
    from matplotlib import cm as cm
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Correlation Titanic')
    labels=list(df.columns.values)
    ax1.set_xticklabels(labels,fontsize=10)
    ax1.set_yticklabels(labels,fontsize=10)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[i/100 for i in range(-55,100,5)])
    plt.show()

def isMale(data):
    if data == "male":
        return 1
    elif data == "female":
        return 0
    else:
        return data

def embarkCodeToInt(embarkCode):
    if embarkCode == 'Q':
        return 1
    elif embarkCode == 'S':
        return 2
    else: return 3

def intToEmbarkCode(integer):
    if integer == 1:
        return 'Q'
    elif integer == 2:
        return 'S'
    else: return 'C'
