import matplotlib.pyplot as plt

plt.style.use("ggplot")

method_color_dict = {
                    "Simplex":"red",
                    "Pontos Interiores":"blue",
                    "Híbrido":"purple"
                    }


def plotvar(var, method, vartype, show=True):
    plt.plot(var, color=method_color_dict[method])
    plt.title(f"{vartype} de {method} em funcao de dimensao")
    if show: plt.show()
