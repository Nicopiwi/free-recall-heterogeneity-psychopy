#%%
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt

#%%
# Cargar el archivo CSV
df_1 = pd.read_csv('./data/1_FreeRecall_2024-09-07_02h06.19.808.csv')
df_2 = pd.read_csv('./data/2_FreeRecall_2024-09-07_15h17.29.714.csv')
df_3 = pd.read_csv('./data/3_FreeRecall_2024-09-09_18h05.34.652.csv')
#%%
categories = {
    'colors': [
        'rojo', 'azul', 'verde', 'amarillo', 'morado', 'negro', 'blanco',
        'gris', 'rosa', 'marron', 'turquesa', 'beige', 'lila', 'granate', 'violeta',
        'ocre', 'lavanda', 'cian', 'dorado', 'celeste'
    ],
    'fruits': [
        'manzana', 'banana', 'uva', 'pera', 'melocoton', 'ciruela', 'cereza',
        'mango', 'kiwi', 'sandia', 'melon', 'pina', 'frutilla', 'granada', 'arandano',
        'frambuesa', 'mora', 'papaya', 'durazno', 'higo', "maracuya", "limon"
    ],
    'countries':  [
      "argentina", "australia", "brasil", "canada", "china", "colombia",
      "espana", "francia", "alemania", "india", "italia", "japon",
      "mexico", "nigeria", "rusia", "sudafrica", "uk", "usa",
      "chile", "peru", "venezuela", "uruguay", "paraguay", "portugal",
      "egipto", "grecia", "suecia", "noruega", "finlandia", "bolivia", "ecuador"
  ],
  'animals': [
      "perro", "gato", "elefante", "leon", "tigre", "jirafa",
      "oso", "lobo", "zorro", "conejo", "ardilla", "canguro",
      "rinoceronte", "hipopotamo", "cebra", "delfin", "ballena", "tiburon",
      "pingüino", "cocodrilo", "aguila", "buho", "colibri", "halcon",
      "murcielago", "camello", "koala", "mono", "panda", "tortuga"
  ],
  "instruments":  [
      "guitarra", "piano", "violin", "flauta", "trompeta", "bateria",
      "saxofon", "clarinete", "trombon", "bajo", "arpa", "ukelele",
      "acordeon", "violonchelo", "contrabajo", "tambor", "mandolina",
      "oboe", "tuba", "xilofono", "armonica", "banjo", "fagot",
      "marimba", "timbal"
  ],
  "verbs":  [
      'comer', 'beber', 'correr', 'saltar', 'hablar', 'leer', 'escribir', 
      'escuchar', 'mirar', 'cocinar', 'viajar', 'pensar', 'dormir', 'caminar', 
      'cantar', 'bailar', 'ensenar', 'aprender', 'jugar', 'trabajar', 
      'abrir', 'cerrar', 'nadar', 'dibujar', 'construir', 'romper', 'pintar', 
      'reir', 'llorar', 'volar'
  ],
  "electro": [
    'lavadora', 'secadora', 'refrigerador', 'microondas', 'licuadora', 
    'batidora', 'tostadora', 'cafetera', 'lavavajillas', 'plancha', 
    'ventilador', 'aspiradora', 'horno', 'calentador', 'extractor', 
    'televisor', 'congelador', 'freidora', 'aire acondicionado', 'deshumidificador', 
    'humidificador', 'secador de pelo', 'radiador', 'estufa', 
    'picadora', 'cortafiambres'
]
  
}
#%%
# Función para estandarizar palabras
def standardize_word(word):
    # Quitar tildes y convertir a minúsculas
    word = ''.join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != 'Mn')
    return word.lower()

# Aplicar función para quitar tildes y convertir a minúsculas
def standardize_words(word_list):
    return [standardize_word(word) for word in word_list]

#%%
# Separar las palabras en las columnas 'words' y 'completion_textbox.text'
dfs = [df_1, df_2]
new_dfs = []

for df in dfs:
    df_new = df[df['trials.thisN'].notna()]
    df_new['words'] = df['words'].astype('str').apply(lambda x: standardize_words(x.split(',')) if type(x) is str else None)
    df_new['completion_textbox.text'] = df['completion_textbox.text'].astype('str').apply(lambda x: standardize_words(x.split('.')) if type(x) is str else None)
    df_new = df_new.loc[:, ('completion_textbox.text', 'words', 'min_category', 'max_category', 'heterogeneity')]
    new_dfs.append(df_new)

df_full = pd.concat(new_dfs)
#%%
# Función para calcular proporción de aciertos en minoritaria/mayoritaria
def calculate_accuracy_min(row):
    if pd.isna(row["min_category"]) or pd.isna(row["max_category"]):
        return None
    
    min_category = row["min_category"]
    min_category_words = set(categories[min_category])
    shown_words = set(row['words'])
    shown_min_category_words = shown_words.intersection(min_category_words)
    user_words = set(row['completion_textbox.text'])
    hits_min = shown_min_category_words.intersection(user_words)

    if len(shown_min_category_words) == 0:
        return 0
    
    return len(hits_min) / len(shown_min_category_words)

def calculate_accuracy_max(row):
    if pd.isna(row["min_category"]) or pd.isna(row["max_category"]):
        return None
    
    max_category = row["max_category"]
    max_category_words = set(categories[max_category])
    shown_words = set(row['words'])
    shown_max_category_words = shown_words.intersection(max_category_words)
    user_words = set(row['completion_textbox.text'])
    hits_max = shown_max_category_words.intersection(user_words)

    
    return len(hits_max) / len(shown_max_category_words)

# Aplicar la función a cada fila
df_full['accuracy_min'] = df_full.apply(calculate_accuracy_min, axis=1)

df_full['accuracy_max'] = df_full.apply(calculate_accuracy_max, axis=1)

# Agrupar por la homogeneidad (asumo que hay una columna 'homogeneity')
df_grouped = df_full.groupby('heterogeneity')['accuracy_min'].mean().reset_index()


# Graficar
plt.figure(figsize=(10,6))
plt.plot(df_grouped['heterogeneity'][1:], df_grouped['accuracy_min'][1:], marker='o')
plt.xlabel('Heterogeneidad')
plt.ylabel('Proporción de aciertos')
plt.title('Proporción de aciertos según heterogeneidad en categoría minoritaria')
plt.grid(True)
plt.show()

# %%
df_grouped_max = df_full.groupby('heterogeneity')['accuracy_max'].mean().reset_index()

plt.figure(figsize=(10,6))
plt.plot(df_grouped_max['heterogeneity'][1:], df_grouped_max['accuracy_max'][1:], marker='o')
plt.xlabel('Heterogeneidad')
plt.ylabel('Proporción de aciertos')
plt.title('Proporción de aciertos según heterogeneidad')
plt.grid(True)
plt.show()
# %%
