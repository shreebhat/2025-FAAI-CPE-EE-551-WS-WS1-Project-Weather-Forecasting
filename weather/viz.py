import matplotlib.pyplot as plt


def dotPlot(x, y):
    plt.plot(x,y, '.' )
    plt.show()



def main() -> None:
    x = [1,2,3,4,5]
    y = [10,15,5,25,20]
    dotPlot(x,y)

if __name__ == "__main__":
    main()