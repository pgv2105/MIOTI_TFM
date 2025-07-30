import json
import matplotlib.pyplot as plt
import heapq


with open("data/monkey_returns.json") as f:
    srs = json.load(f)

real_sr = 39.96  # tu retorno real


n_mayores = heapq.nlargest(5, srs)  # 3 mayores valores
print(n_mayores) 

plt.figure(figsize=(8,5))
plt.hist(srs, bins=50, alpha=0.7, color='blue', edgecolor='white')
plt.axvline(real_sr, color='red', linewidth=2, label=f"Real return = {real_sr:.4f} %")
plt.title("Proporción de retornos: Monos vs. Modelo")
plt.xlabel("Retornos (%)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.show()
