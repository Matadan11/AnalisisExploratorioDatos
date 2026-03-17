# Paquete: Analisis de Datos Exploratorios (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from abc import ABC, abstractmethod

# Modelos No Supervisados
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Utilidades para HAC (dendrogramas)
from scipy.cluster.hierarchy import linkage, dendrogram

class AnalisisEDA:

    def __init__(self, path=None, sep=",", decimal=".", index_col=None, nrows=None, df=None):
        if df is not None:
            self.__df = df.copy()
        elif path is not None:
            self.__df = self.__datosCargados(path, sep=sep, decimal=decimal, index_col=index_col, nrows=nrows)
        else:
            self.__df = pd.DataFrame()

    # Getter / Setter del DataFrame
    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, p_df):
        self.__df = p_df

    # Helpers
    def __validarDF(self):
        if self.__df is None or self.__df.empty:
            raise ValueError("El DataFrame esta vacio. Carga un dataset primero con cargarCSV() o pasando df=...")

    def __columnasNumericas(self):
        self.__validarDF()
        return self.__df.select_dtypes(include=["number"]).columns.tolist()

    def __columnasCategoricas(self):
        self.__validarDF()
        return self.__df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Cargar el dataset
    def __datosCargados(self, path, sep=",", decimal=".", index_col=None, nrows=None):
        return pd.read_csv(
            path,
            sep=sep,
            decimal=decimal,
            index_col=index_col,
            nrows=nrows
        )

    def cargarCSV(self, path, sep=",", decimal=".", index_col=None, nrows=None):
        self.__df = self.__datosCargados(path, sep=sep, decimal=decimal, index_col=index_col, nrows=nrows)
        print(f"Dataset cargado. Dimensiones: {self.__df.shape}")
        return self.__df

    # Verificar tipos de datos
    def tipoDatos(self):
        self.__validarDF()
        print("Tipos de datos por columna:\n")
        print(self.__df.dtypes)

    # Resumen general
    def analisisGeneral(self, head=5):
        self.__validarDF()
        print("Dimensiones:", self.__df.shape)
        print("\nHead:\n")
        print(self.__df.head(head))

        print("\n" + "=" * 40)
        print("Estadisticas Descriptivas (numericas)")
        print("=" * 40)
        try:
            print(self.__df.describe(include=[np.number]).T)
        except Exception:
            print(self.__df.describe().T)

        return self.__df

    # Eliminar columnas irrelevantes
    def eliminarColumnas(self, columnas, inplace=True):
        self.__validarDF()
        if columnas is None or len(columnas) == 0:
            print("No se pasaron columnas para eliminar.")
            return self.__df

        existentes = [c for c in columnas if c in self.__df.columns]
        no_existen = [c for c in columnas if c not in self.__df.columns]

        if len(no_existen) > 0:
            print(f"Columnas que no existen (se ignoran): {no_existen}")

        if len(existentes) == 0:
            print("Ninguna de las columnas indicadas existe en el DataFrame.")
            return self.__df

        if inplace:
            self.__df.drop(columns=existentes, inplace=True)
            print(f"Columnas eliminadas: {existentes}")
            print(f"Dimensiones actuales: {self.__df.shape}")
            return self.__df
        else:
            df_nuevo = self.__df.drop(columns=existentes)
            print(f"Columnas eliminadas (sin modificar original): {existentes}")
            print(f"Dimensiones nuevas: {df_nuevo.shape}")
            return df_nuevo

    # Renombrar columnas
    def renombrarColumnas(self, nuevos_nombres, inplace=True):
        self.__validarDF()
        if nuevos_nombres is None or len(nuevos_nombres) == 0:
            print("No se paso diccionario de renombre.")
            return self.__df

        if inplace:
            self.__df.rename(columns=nuevos_nombres, inplace=True)
            print("Columnas renombradas.")
            return self.__df
        else:
            df_nuevo = self.__df.rename(columns=nuevos_nombres)
            print("Columnas renombradas (sin modificar original).")
            return df_nuevo

    # Eliminar filas duplicadas
    def eliminarDuplicados(self, inplace=True, keep="first"):
        self.__validarDF()
        antes = self.__df.shape[0]
        df_sin = self.__df.drop_duplicates(keep=keep)
        despues = df_sin.shape[0]
        print(f"Se eliminaron {antes - despues} filas duplicadas. Total actual: {despues} filas.")

        if inplace:
            self.__df = df_sin
            return self.__df
        return df_sin

    # Valores faltantes / nulos
    def valores_faltantes(self):
        self.__validarDF()
        missing = self.__df.isna().sum()
        print("Missing values por columna:\n")
        print(missing)
        return missing

    def eliminarNulos(self, inplace=True, how="any", subset=None):
        self.__validarDF()

        nulos_antes = int(self.__df.isnull().sum().sum())
        filas_antes = self.__df.shape[0]

        df_sin = self.__df.dropna(how=how, subset=subset)

        nulos_despues = int(df_sin.isnull().sum().sum())
        filas_despues = df_sin.shape[0]

        print(f"Valores nulos totales antes: {nulos_antes}")
        print(f"Filas eliminadas por nulos: {filas_antes - filas_despues}")
        print(f"Valores nulos restantes: {nulos_despues}")

        if inplace:
            self.__df = df_sin
            return self.__df
        return df_sin

    # Valores unicos
    def valores_unicos(self, columna):
        self.__validarDF()
        if columna not in self.__df.columns:
            print(f"La columna '{columna}' no existe.")
            return None

        vc = self.__df[columna].value_counts(dropna=False)
        print(f"Valores unicos en '{columna}':\n")
        print(vc)
        return vc


    # Convertir categóricas a dummies (one-hot)
    def categoricasADummies(self, columnas=None, drop_first=True, dummy_na=False, dtype=int, inplace=True, excluir=None):
        self.__validarDF()

        if excluir is None:
            excluir = []

        if columnas is None:
            columnas = self.__columnasCategoricas()

        columnas = [c for c in columnas if (c in self.__df.columns) and (c not in excluir)]

        if len(columnas) == 0:
            print("No hay columnas categóricas para convertir a dummies (o fueron excluidas).")
            return self.__df

        print(f"Columnas categóricas convertidas a dummies: {columnas}")

        df_nuevo = pd.get_dummies(
            self.__df,
            columns=columnas,
            drop_first=drop_first,
            dummy_na=dummy_na,
            dtype=dtype
        )

        if inplace:
            self.__df = df_nuevo
            print(f"Dimensiones actuales: {self.__df.shape}")
            return self.__df

        print(f"Dimensiones nuevas: {df_nuevo.shape}")
        return df_nuevo

    # Detectar valores atipicos (IQR)
    def detectarAtipicosIQR(self, columnas=None, factor=1.5):
        self.__validarDF()

        if columnas is None:
            columnas = self.__columnasNumericas()
        else:
            columnas = [c for c in columnas if c in self.__df.columns]

        if len(columnas) == 0:
            print("No hay columnas numericas para detectar atipicos.")
            return pd.DataFrame()

        filas_total = self.__df.shape[0]
        resumen = []

        for col in columnas:
            serie = self.__df[col].dropna()
            if serie.empty:
                continue

            q1 = float(serie.quantile(0.25))
            q3 = float(serie.quantile(0.75))
            iqr = q3 - q1
            lim_inf = q1 - factor * iqr
            lim_sup = q3 + factor * iqr

            mask_out = (self.__df[col] < lim_inf) | (self.__df[col] > lim_sup)
            n_out = int(mask_out.sum())
            pct_out = (n_out / filas_total) * 100 if filas_total > 0 else 0

            resumen.append([col, q1, q3, iqr, lim_inf, lim_sup, n_out, pct_out])

        df_resumen = pd.DataFrame(
            resumen,
            columns=["columna", "q1", "q3", "iqr", "lim_inf", "lim_sup", "n_atipicos", "pct_atipicos"]
        )

        print("Resumen de atipicos (IQR):\n")
        print(df_resumen.sort_values("n_atipicos", ascending=False))
        return df_resumen

    def eliminarAtipicosIQR(self, columnas=None, factor=1.5, inplace=True):
        self.__validarDF()

        if columnas is None:
            columnas = self.__columnasNumericas()
        else:
            columnas = [c for c in columnas if c in self.__df.columns]

        if len(columnas) == 0:
            print("No hay columnas numericas para eliminar atipicos.")
            return self.__df

        mask_keep = np.ones(len(self.__df), dtype=bool)

        for col in columnas:
            serie = self.__df[col].dropna()
            if serie.empty:
                continue

            q1 = float(serie.quantile(0.25))
            q3 = float(serie.quantile(0.75))
            iqr = q3 - q1
            lim_inf = q1 - factor * iqr
            lim_sup = q3 + factor * iqr

            mask_keep &= ~((self.__df[col] < lim_inf) | (self.__df[col] > lim_sup))

        antes = self.__df.shape[0]
        df_filtrado = self.__df.loc[mask_keep].copy()
        despues = df_filtrado.shape[0]

        print(f"Filas eliminadas por atipicos (IQR): {antes - despues}")
        print(f"Total actual: {despues} filas")

        if inplace:
            self.__df = df_filtrado
            return self.__df
        return df_filtrado

    # 9) Graficos
    def graficoBoxplot(self, columnas=None, ncols=3, dpi=150):
        self.__validarDF()

        if columnas is None:
            columnas = self.__columnasNumericas()
        else:
            columnas = [c for c in columnas if c in self.__df.columns]

        n = len(columnas)
        if n == 0:
            print("No hay variables numericas para boxplot.")
            return

        filas = math.ceil(n / ncols)
        fig, axes = plt.subplots(filas, ncols, figsize=(5 * ncols, 4 * filas), dpi=dpi)
        axes = np.array(axes).flatten()

        for i, col in enumerate(columnas):
            sns.boxplot(y=self.__df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot de {col}", fontsize=10)
            axes[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def histogramas(self, columnas=None, bins=30, ncols=3, dpi=150):
        self.__validarDF()

        if columnas is None:
            columnas = self.__columnasNumericas()
        else:
            columnas = [c for c in columnas if c in self.__df.columns]

        n = len(columnas)
        if n == 0:
            print("No hay variables numericas para histogramas.")
            return

        filas = math.ceil(n / ncols)
        fig, axes = plt.subplots(filas, ncols, figsize=(5 * ncols, 4 * filas), dpi=dpi)
        axes = np.array(axes).flatten()

        for i, col in enumerate(columnas):
            axes[i].hist(self.__df[col].dropna(), bins=bins, edgecolor="black", alpha=0.7)
            axes[i].set_title(f"Histograma de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def distribucionVariables(self, columnas=None, bins=30, kde=True, ncols=3, dpi=150):
        self.__validarDF()

        if columnas is None:
            columnas = self.__columnasNumericas()
        else:
            columnas = [c for c in columnas if c in self.__df.columns]

        n = len(columnas)
        if n == 0:
            print("No hay variables numericas para distribucion.")
            return

        filas = math.ceil(n / ncols)
        fig, axes = plt.subplots(filas, ncols, figsize=(5 * ncols, 4 * filas), dpi=dpi)
        axes = np.array(axes).flatten()

        for i, col in enumerate(columnas):
            sns.histplot(self.__df[col].dropna(), bins=bins, kde=kde, ax=axes[i])
            axes[i].set_title(f"Distribucion de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def datosDensidad(self, columnas=None, ncols=3, dpi=150):
        self.__validarDF()

        if columnas is None:
            columnas = self.__columnasNumericas()
        else:
            columnas = [c for c in columnas if c in self.__df.columns]

        n = len(columnas)
        if n == 0:
            print("No hay variables numericas para densidad.")
            return

        filas = math.ceil(n / ncols)
        fig, axes = plt.subplots(filas, ncols, figsize=(5 * ncols, 4 * filas), dpi=dpi)
        axes = np.array(axes).flatten()

        for i, col in enumerate(columnas):
            sns.kdeplot(x=self.__df[col].dropna(), fill=True, ax=axes[i], linewidth=2)
            axes[i].set_title(f"Densidad de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Densidad")
            axes[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def correlaciones(self, method="pearson"):
        self.__validarDF()
        corr = self.__df.corr(numeric_only=True, method=method)
        print("Matriz de correlaciones:\n")
        print(corr)
        return corr

    def graficoCorrelacion(self, method="pearson", annot=True, fmt=".2f", dpi=150):
        self.__validarDF()
        corr = self.__df.corr(numeric_only=True, method=method)

        plt.figure(figsize=(12, 8), dpi=dpi)
        sns.heatmap(
            corr,
            vmin=-1, vmax=1,
            annot=annot, fmt=fmt,
            linewidths=0.5, linecolor="white",
            square=True
        )
        plt.title("Mapa de Calor de Correlaciones", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return corr

    def graficosDispersion(self, columnas=None):
        self.__validarDF()

        if columnas is None:
            columnas = self.__columnasNumericas()
        else:
            columnas = [c for c in columnas if c in self.__df.columns]

        if len(columnas) < 2:
            print("No hay suficientes variables numericas para graficar dispersion.")
            return

        sns.pairplot(self.__df[columnas].dropna())
        plt.suptitle("Graficos de Dispersion por Pares", y=1.02)
        plt.show()

    def histogramaClase(self, columna_objetivo, dpi=150):
        self.__validarDF()
        if columna_objetivo not in self.__df.columns:
            print(f"La columna '{columna_objetivo}' no existe en el DataFrame.")
            return

        plt.figure(figsize=(8, 5), dpi=dpi)
        self.__df[columna_objetivo].value_counts(dropna=False).plot(kind="bar")
        plt.title(f"Distribucion de la Clase: {columna_objetivo}")
        plt.xlabel(columna_objetivo)
        plt.ylabel("Frecuencia")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

class NoSupervisado(AnalisisEDA):
    def __init__(self, path=None, sep=",", decimal=".", index_col=None, nrows=None, df=None):
        super().__init__(path=path, sep=sep, decimal=decimal, index_col=index_col, nrows=nrows, df=df)

        # Cache de la última matriz usada (útil para repetir experimentos sin recalcular)
        self.__X_cache = None
        self.__X_index = None
        self.__feature_names = None
        self.__scaler = None
        self.__X_config_cache = None  # (columnas_tuple, escalar, dropna)

        # Últimos resultados
        self.__acp_resultado = None
        self.__tsne_resultado = None
        self.__umap_resultado = None

    # Helpers No Supervisados

    def __validarDF_NS(self):
        if self.df is None or self.df.empty:
            raise ValueError("El DataFrame esta vacio. Carga un dataset primero con cargarCSV() o pasando df=...")

    def matrizX(self, columnas=None, escalar=True, dropna=True, forzar_recalculo=False):
        self.__validarDF_NS()

        if columnas is None:
            columnas = self.df.select_dtypes(include=["number"]).columns.tolist()
        else:
            columnas = [c for c in columnas if c in self.df.columns]

        if len(columnas) == 0:
            raise ValueError("No hay columnas numéricas para construir la matriz X.")

        config = (tuple(columnas), bool(escalar), bool(dropna))
        if (not forzar_recalculo) and (self.__X_cache is not None) and (self.__X_config_cache == config):
            return self.__X_cache

        dfX = self.df[columnas].copy()

        if dropna:
            dfX = dfX.dropna(axis=0, how="any")

        self.__X_index = dfX.index.copy()
        self.__feature_names = list(dfX.columns)

        X = dfX.values

        if escalar:
            self.__scaler = StandardScaler()
            X = self.__scaler.fit_transform(X)
        else:
            self.__scaler = None

        self.__X_cache = X
        self.__X_config_cache = config

        print(f"Matriz X lista. shape={X.shape} | escalar={escalar} | dropna={dropna} | cols={len(columnas)}")
        return X

    def infoMatrizX(self, max_cols=25):
        if self.__X_cache is None:
            print("Aún no has construido X. Llama primero a matrizX().")
            return

        print(f"X shape: {self.__X_cache.shape}")
        print(f"Escalado: {'SI' if self.__scaler is not None else 'NO'}")
        cols = self.__feature_names if self.__feature_names is not None else []
        if len(cols) <= max_cols:
            print(f"Columnas ({len(cols)}): {cols}")
        else:
            print(f"Columnas ({len(cols)}): {cols[:max_cols]} ...")

    def plotEmbedding(self, embedding, hue=None, titulo="Embedding 2D", dpi=150, alpha=0.8):
        if embedding is None:
            print("No hay embedding para graficar.")
            return

        emb = np.array(embedding)
        if emb.ndim != 2 or emb.shape[1] < 2:
            raise ValueError("El embedding debe ser 2D con al menos 2 columnas.")

        colores = None
        if hue is not None:
            if isinstance(hue, str) and hue in self.df.columns:
                if self.__X_index is not None:
                    colores = self.df.loc[self.__X_index, hue].values
                else:
                    colores = self.df[hue].values
            else:
                colores = np.array(hue)

        plt.figure(figsize=(8, 6), dpi=dpi)
        if colores is None:
            plt.scatter(emb[:, 0], emb[:, 1], alpha=alpha)
        else:
            plt.scatter(emb[:, 0], emb[:, 1], c=colores, alpha=alpha)
            plt.colorbar()

        plt.title(titulo)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    # ACP / PCA 

    def acp(self, n_componentes=2, columnas=None, escalar=True, dropna=True, svd_solver="auto", random_state=42):

        X = self.matrizX(columnas=columnas, escalar=escalar, dropna=dropna)

        pca = PCA(n_components=n_componentes, svd_solver=svd_solver, random_state=random_state)
        Z = pca.fit_transform(X)

        var_ratio = pca.explained_variance_ratio_
        var_acum = np.cumsum(var_ratio)

        df_comp = pd.DataFrame(
            Z,
            index=self.__X_index,
            columns=[f"PC{i+1}" for i in range(Z.shape[1])]
        )

        resultado = {
            "modelo": pca,
            "componentes": Z,
            "df_componentes": df_comp,
            "var_ratio": var_ratio,
            "var_acum": var_acum,
            "columnas": self.__feature_names,
            "escalar": escalar,
        }

        self.__acp_resultado = resultado

        print(f"ACP listo. n_componentes={n_componentes} | var_exp_total={var_acum[-1]:.4f}")
        return resultado

    def compararACP(self, n_componentes_lista=(2, 3, 5, 8, 10), columnas=None, escalar=True, dropna=True, var_objetivo=None):
        rows = []
        for n in n_componentes_lista:
            try:
                res = self.acp(n_componentes=n, columnas=columnas, escalar=escalar, dropna=dropna)
                rows.append([n, float(res["var_acum"][-1])])
            except Exception as e:
                rows.append([n, np.nan])
                print(f"ACP falló para n={n}: {e}")

        tabla = pd.DataFrame(rows, columns=["n_componentes", "varianza_acumulada"])
        tabla = tabla.sort_values("n_componentes").reset_index(drop=True)

        sugerido = None
        if var_objetivo is not None:
            cand = tabla.dropna().query("varianza_acumulada >= @var_objetivo")
            if not cand.empty:
                sugerido = int(cand.iloc[0]["n_componentes"])

        print("Comparación ACP:")
        print(tabla)

        if var_objetivo is not None:
            if sugerido is None:
                print(f"No se alcanzó var_objetivo={var_objetivo} con los n_componentes probados.")
            else:
                print(f"Sugerencia: n_componentes={sugerido} (>= {var_objetivo} de varianza acumulada)")

        return tabla, sugerido

    def graficoVarianzaACP(self, resultado=None, dpi=150):
        if resultado is None:
            resultado = self.__acp_resultado

        if resultado is None:
            print("No hay resultado ACP. Ejecuta acp() primero.")
            return

        var_ratio = np.array(resultado["var_ratio"])
        var_acum = np.array(resultado["var_acum"])
        x = np.arange(1, len(var_ratio) + 1)

        plt.figure(figsize=(9, 5), dpi=dpi)
        plt.bar(x, var_ratio, alpha=0.8, label="Varianza por componente")
        plt.plot(x, var_acum, marker="o", label="Varianza acumulada")
        plt.xticks(x)
        plt.xlabel("Componente")
        plt.ylabel("Proporción de varianza")
        plt.title("ACP/PCA - Varianza explicada")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plotPlanoACP(self, resultado=None, ejes=(1, 2), hue=None, titulo="Plano ACP", dpi=150):
        if resultado is None:
            resultado = self.__acp_resultado

        if resultado is None:
            print("No hay resultado ACP. Ejecuta acp() primero.")
            return

        Z = np.array(resultado["componentes"])
        e0 = int(ejes[0]) - 1
        e1 = int(ejes[1]) - 1

        if e0 < 0 or e1 < 0 or e0 >= Z.shape[1] or e1 >= Z.shape[1]:
            raise ValueError("Ejes fuera de rango para el número de componentes calculados.")

        emb = Z[:, [e0, e1]]
        self.plotEmbedding(emb, hue=hue, titulo=titulo, dpi=dpi)

    def plotCirculoCorrelacionACP(self, resultado=None, ejes=(1, 2), top_n=None, dpi=150, titulo="Círculo de correlación (aprox.)"):
        if resultado is None:
            resultado = self.__acp_resultado

        if resultado is None:
            print("No hay resultado ACP. Ejecuta acp() primero.")
            return

        pca = resultado["modelo"]
        columnas = resultado["columnas"]
        e0 = int(ejes[0]) - 1
        e1 = int(ejes[1]) - 1

        # loadings aproximados
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        coords = loadings[:, [e0, e1]]
        mags = np.sqrt((coords ** 2).sum(axis=1))

        idx = np.arange(coords.shape[0])
        if top_n is not None and top_n < len(idx):
            idx = np.argsort(mags)[::-1][:top_n]

        plt.figure(figsize=(7, 7), dpi=dpi)
        circle = plt.Circle((0, 0), 1, color="steelblue", fill=False)
        plt.gca().add_patch(circle)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.6)
        plt.axvline(0, color="gray", linestyle="--", alpha=0.6)

        for i in idx:
            x, y = coords[i, 0], coords[i, 1]
            plt.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, alpha=0.6, length_includes_head=True)
            plt.text(x * 1.08, y * 1.08, str(columnas[i]), ha="center", va="center", fontsize=9)

        plt.title(titulo)
        plt.xlabel(f"PC{e0+1}")
        plt.ylabel(f"PC{e1+1}")
        plt.axis("scaled")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # t-SNE

    def tsne(self, n_componentes=2, columnas=None, escalar=True, dropna=True,
             perplexity=30, max_iter=1000, init="pca", learning_rate="auto", random_state=42):
        """t-SNE para visualización (embedding)."""
        X = self.matrizX(columnas=columnas, escalar=escalar, dropna=dropna)

        tsne = TSNE(
            n_components=n_componentes,
            perplexity=perplexity,
            max_iter=max_iter,
            init=init,
            learning_rate=learning_rate,
            random_state=random_state
        )
        Z = tsne.fit_transform(X)

        resultado = {
            "modelo": tsne,
            "embedding": Z,
            "perplexity": perplexity,
            "max_iter": max_iter,
            "escalar": escalar,
            "columnas": self.__feature_names
        }
        self.__tsne_resultado = resultado

        print(f"t-SNE listo. perplexity={perplexity} | max_iter={max_iter} | shape={Z.shape}")
        return resultado

    def evaluarTrustworthiness(self, embedding, n_neighbors=5):
        if self.__X_cache is None:
            print("No hay X en cache. Llama primero a matrizX() o a tsne()/umap().")
            return None

        emb = np.array(embedding)
        score = trustworthiness(self.__X_cache, emb, n_neighbors=n_neighbors)
        print(f"Trustworthiness (n_neighbors={n_neighbors}): {score:.4f}")
        return float(score)

    def compararTSNE(self, configuraciones, columnas=None, escalar=True, dropna=True, n_neighbors_trust=5):
        tabla = []
        for cfg in configuraciones:
            res = self.tsne(columnas=columnas, escalar=escalar, dropna=dropna, **cfg)
            tw = self.evaluarTrustworthiness(res["embedding"], n_neighbors=n_neighbors_trust)
            fila = {
                "perplexity": res["perplexity"],
                "max_iter": res["max_iter"],
                "trustworthiness": tw
            }
            tabla.append(fila)

        df_tabla = pd.DataFrame(tabla).sort_values("trustworthiness", ascending=False).reset_index(drop=True)
        print("Comparación t-SNE:")
        print(df_tabla)
        return df_tabla


    # UMAP

    def umap(self, n_componentes=2, columnas=None, escalar=True, dropna=True,
             n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42):
        try:
            import umap
        except Exception as e:
            print("UMAP no está instalado. Instala con: pip install umap-learn")
            print(f"Detalle: {e}")
            return None

        X = self.matrizX(columnas=columnas, escalar=escalar, dropna=dropna)

        reducer = umap.UMAP(
            n_components=n_componentes,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        Z = reducer.fit_transform(X)

        resultado = {
            "modelo": reducer,
            "embedding": Z,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "escalar": escalar,
            "columnas": self.__feature_names
        }
        self.__umap_resultado = resultado

        print(f"UMAP listo. n_neighbors={n_neighbors} | min_dist={min_dist} | shape={Z.shape}")
        return resultado

    def compararUMAP(self, configuraciones, columnas=None, escalar=True, dropna=True, n_neighbors_trust=5):
        tabla = []
        for cfg in configuraciones:
            res = self.umap(columnas=columnas, escalar=escalar, dropna=dropna, **cfg)
            if res is None:
                continue
            tw = self.evaluarTrustworthiness(res["embedding"], n_neighbors=n_neighbors_trust)
            fila = {
                "n_neighbors": res["n_neighbors"],
                "min_dist": res["min_dist"],
                "metric": res["metric"],
                "trustworthiness": tw
            }
            tabla.append(fila)

        df_tabla = pd.DataFrame(tabla).sort_values("trustworthiness", ascending=False).reset_index(drop=True)
        print("Comparación UMAP:")
        print(df_tabla)
        return df_tabla


class Clustering(NoSupervisado):

    def __init__(self, path=None, sep=",", decimal=".", index_col=None, nrows=None, df=None):
        super().__init__(path=path, sep=sep, decimal=decimal, index_col=index_col, nrows=nrows, df=df)

        self.__kmeans_resultado = None
        self.__hac_resultado = None


    # KMeans (Kmedias)

    def kmedias(self, n_clusters=3, columnas=None, escalar=True, dropna=True,
                init="k-means++", n_init=10, max_iter=300, random_state=42):
        X = self.matrizX(columnas=columnas, escalar=escalar, dropna=dropna)

        modelo = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        labels = modelo.fit_predict(X)

        resultado = {
            "modelo": modelo,
            "labels": labels,
            "inercia": float(modelo.inertia_),
            "n_clusters": n_clusters,
            "escalar": escalar
        }
        self.__kmeans_resultado = resultado

        print(f"KMeans listo. k={n_clusters} | inercia={modelo.inertia_:.2f}")
        return resultado

    def evaluarKmedias(self, k_min=2, k_max=10, columnas=None, escalar=True, dropna=True, random_state=42):
        X = self.matrizX(columnas=columnas, escalar=escalar, dropna=dropna)
        filas = []
        for k in range(int(k_min), int(k_max) + 1):
            modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = modelo.fit_predict(X)

            sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
            db = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
            ch = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan

            filas.append([k, float(modelo.inertia_), float(sil) if not np.isnan(sil) else np.nan,
                          float(db) if not np.isnan(db) else np.nan,
                          float(ch) if not np.isnan(ch) else np.nan])

        tabla = pd.DataFrame(filas, columns=["k", "inercia", "silhouette", "davies_bouldin", "calinski_harabasz"])
        tabla = tabla.sort_values("k").reset_index(drop=True)

        # mejor por silhouette (max)
        mejor_k = None
        if tabla["silhouette"].notna().any():
            mejor_k = int(tabla.iloc[tabla["silhouette"].idxmax()]["k"])

        print("Evaluación KMeans:")
        print(tabla)
        if mejor_k is not None:
            print(f"Mejor k (por silhouette): {mejor_k}")

        return tabla, mejor_k

    def graficoCodoSilhouette(self, tabla, dpi=150):
        if tabla is None or len(tabla) == 0:
            print("Tabla vacía.")
            return

        k = tabla["k"].values
        iner = tabla["inercia"].values
        sil = tabla["silhouette"].values

        plt.figure(figsize=(9, 4), dpi=dpi)
        plt.plot(k, iner, marker="o")
        plt.title("KMeans - Codo (Inercia)")
        plt.xlabel("k")
        plt.ylabel("Inercia")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(9, 4), dpi=dpi)
        plt.plot(k, sil, marker="o")
        plt.title("KMeans - Silhouette vs k")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    # HAC (Agglomerative)

    def hac(self, n_clusters=3, columnas=None, escalar=True, dropna=True, linkage_="ward", metric="euclidean"):
        X = self.matrizX(columnas=columnas, escalar=escalar, dropna=dropna)

        # Restricción de ward
        if linkage_ == "ward" and metric != "euclidean":
            print("linkage='ward' requiere metric='euclidean'. Se ajusta metric='euclidean'.")
            metric = "euclidean"

        try:
            modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_, metric=metric)
        except TypeError:
            # compatibilidad con versiones antiguas de scikit-learn
            modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_, affinity=metric)

        labels = modelo.fit_predict(X)

        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
        db = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
        ch = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan

        resultado = {
            "modelo": modelo,
            "labels": labels,
            "n_clusters": n_clusters,
            "linkage": linkage_,
            "metric": metric,
            "silhouette": float(sil) if not np.isnan(sil) else np.nan,
            "davies_bouldin": float(db) if not np.isnan(db) else np.nan,
            "calinski_harabasz": float(ch) if not np.isnan(ch) else np.nan,
            "escalar": escalar
        }
        self.__hac_resultado = resultado

        print(f"HAC listo. k={n_clusters} | linkage={linkage_} | metric={metric} | silhouette={sil:.4f}")
        return resultado

    def evaluarHAC(self, n_clusters_lista=(2, 3, 4, 5), linkage_lista=("ward", "complete", "average"),
                   metric_lista=("euclidean", "manhattan"), columnas=None, escalar=True, dropna=True):
        filas = []
        for k in n_clusters_lista:
            for link in linkage_lista:
                for met in metric_lista:
                    try:
                        res = self.hac(n_clusters=int(k), linkage_=link, metric=met, columnas=columnas, escalar=escalar, dropna=dropna)
                        filas.append([k, link, res["metric"], res["silhouette"], res["davies_bouldin"], res["calinski_harabasz"]])
                    except Exception as e:
                        print(f"HAC falló k={k}, linkage={link}, metric={met}: {e}")

        tabla = pd.DataFrame(
            filas,
            columns=["k", "linkage", "metric", "silhouette", "davies_bouldin", "calinski_harabasz"]
        )

        if len(tabla) == 0:
            print("No se pudo construir tabla HAC.")
            return tabla, None

        tabla = tabla.sort_values("silhouette", ascending=False).reset_index(drop=True)

        mejor = tabla.iloc[0].to_dict()
        print("Evaluación HAC (ordenado por silhouette):")
        print(tabla)
        print(f"Mejor configuración HAC: {mejor}")

        return tabla, mejor

    def dendrogramaHAC(self, metodo="ward", metric="euclidean", columnas=None, escalar=True, dropna=True,
                       truncate_mode=None, p=30, dpi=150):
        X = self.matrizX(columnas=columnas, escalar=escalar, dropna=dropna)

        if metodo == "ward" and metric != "euclidean":
            print("metodo='ward' requiere metric='euclidean'. Se ajusta metric='euclidean'.")
            metric = "euclidean"

        Z = linkage(X, method=metodo, metric=metric)

        plt.figure(figsize=(12, 6), dpi=dpi)
        dendrogram(Z, truncate_mode=truncate_mode, p=p)
        plt.title(f"Dendrograma HAC | metodo={metodo} | metric={metric}")
        plt.xlabel("Observaciones")
        plt.ylabel("Distancia")
        plt.tight_layout()
        plt.show()

        return Z

    # Utilidades
    def agregarClustersDF(self, labels, nombre_col="cluster", inplace=True):
        """Agrega los labels al df (se alinea con el index usado en X si dropna=True)."""
        if self.df is None or self.df.empty:
            raise ValueError("El DataFrame esta vacio. Carga un dataset primero con cargarCSV() o pasando df=...")

        idx = getattr(self, "_NoSupervisado__X_index", None)

        if idx is not None:
            serie = pd.Series(labels, index=idx, name=nombre_col)
            if inplace:
                self.df.loc[serie.index, nombre_col] = serie.values
                print(f"Columna '{nombre_col}' agregada (alineada a X).")
                return self.df
            else:
                df_nuevo = self.df.copy()
                df_nuevo.loc[serie.index, nombre_col] = serie.values
                print(f"Columna '{nombre_col}' agregada (sin modificar original).")
                return df_nuevo
        else:
            if inplace:
                self.df[nombre_col] = labels
                print(f"Columna '{nombre_col}' agregada.")
                return self.df
            else:
                df_nuevo = self.df.copy()
                df_nuevo[nombre_col] = labels
                print(f"Columna '{nombre_col}' agregada (sin modificar original).")
                return df_nuevo

    def resumenClusters(self, columna_cluster="cluster"):
        if self.df is None or self.df.empty:
            raise ValueError("El DataFrame esta vacio. Carga un dataset primero con cargarCSV() o pasando df=...")

        if columna_cluster not in self.df.columns:
            print(f"No existe la columna '{columna_cluster}'.")
            return None

        num_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        if columna_cluster in num_cols:
            num_cols.remove(columna_cluster)

        resumen = self.df.groupby(columna_cluster)[num_cols].mean()
        conteo = self.df[columna_cluster].value_counts().sort_index()

        print("Tamaño por cluster:")
        print(conteo)

        print("\nMedias numéricas por cluster:")
        print(resumen)

        return {"conteo": conteo, "medias": resumen}


#class Supervisado(AnalisisEDA):

#class Clasificacion(Supervisado):

#class Regresion(Supervisado):
