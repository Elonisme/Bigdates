import matplotlib.pyplot as plt


class Boxplots:
    def plot(selfs):
        date = [-35,10,20,30,40,50,60,106]
        filerprops = {'marker':'o','markerfacecolor':'red','color':'black'}
        plt.grid(True, linestyle = "-.",color = "black",linewidth = "0.4")
        plt.boxplot(date,notch = False, flierprops = filerprops)
        plt.show()


if __name__ == '__main__':
    B =Boxplots()
    B.plot()

    