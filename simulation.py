import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# 1. НАСТРОЙКИ СИМУЛЯЦИИ
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}")

GRID_SIZE = 128
TIME_STEPS = 2000
DT = 0.1
DX = 1.0
D_COEFF = 0.1

# Проверка устойчивости
stability = DT * D_COEFF / (DX * DX)
print(f"Параметр устойчивости: {stability:.4f} (должно быть <= 0.25)")
if stability > 0.25:
    print("ВНИМАНИЕ: Схема может быть нестабильна!")

# 2. ИНИЦИАЛИЗАЦИЯ
rho = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE)
center = GRID_SIZE // 2
rho[center-1:center+2, center-1:center+2] = 10.0

# Массив для хранения истории максимумов
max_history = []

# 3. СОЗДАНИЕ ТРЁХ ПАНЕЛЕЙ
fig = plt.figure(figsize=(18, 6))

# Левая панель: основная симуляция
ax_main = plt.subplot(1, 3, 1)
cax = ax_main.imshow(rho.cpu().numpy(), cmap='magma', vmin=0, vmax=10.0)
ax_main.set_title("Симуляция диффузии")
plt.colorbar(cax, ax=ax_main)

# Средняя панель: график Max(rho)
ax_graph = plt.subplot(1, 3, 2)
ax_graph.set_title("Максимальная плотность vs Время")
ax_graph.set_xlabel("Шаг")
ax_graph.set_ylabel("Max ρ")
ax_graph.grid(True)
line_graph, = ax_graph.plot([], [], 'b-', linewidth=2)

# Правая панель: текстовое описание
ax_text = plt.subplot(1, 3, 3)
ax_text.axis('off')

text_content = """
УРАВНЕНИЕ ДИФФУЗИИ:

∂ρ/∂t = D·∇²ρ

где:
• ρ - плотность
• D - коэффициент диффузии
• ∇² - оператор Лапласа

ЧИСЛЕННАЯ СХЕМА:

∇²ρ ≈ (ρ[i+1,j] + ρ[i-1,j] + 
       ρ[i,j+1] + ρ[i,j-1] - 4ρ[i,j]) / Δx²

УСЛОВИЕ УСТОЙЧИВОСТИ:

Δt·D/Δx² ≤ 0.25

ПРИНЦИП МАКСИМУМА:

max(ρ) монотонно убывает → 
сингулярности невозможны

ПАРАМЕТРЫ:
• DT = 0.1
• D = 0.1  
• Δx = 1.0
• Сетка: 128×128
"""

ax_text.text(0.05, 0.95, text_content, 
             fontsize=10, verticalalignment='top',
             fontfamily='monospace')

print(f"Начало: Max = {rho.max().item():.4f}")

# 4. ФУНКЦИЯ ОБНОВЛЕНИЯ
def update(frame):
    global rho
    
    # Лапласиан (5-точечный шаблон)
    laplacian = (
        torch.roll(rho, -1, dims=0) +
        torch.roll(rho, 1, dims=0) +
        torch.roll(rho, -1, dims=1) +
        torch.roll(rho, 1, dims=1) -
        4 * rho
    ) / (DX * DX)
    
    # Уравнение диффузии
    rho = rho + DT * D_COEFF * laplacian
    rho = torch.clamp(rho, min=0.0)
    
    # Сохраняем максимум
    current_max = rho.max().item()
    max_history.append(current_max)
    
    # Обновляем основную панель
    cax.set_array(rho.cpu().numpy())
    ax_main.set_title(f"Шаг {frame} | Max: {current_max:.4f}")
    
    # Обновляем график
    line_graph.set_data(range(len(max_history)), max_history)
    ax_graph.relim()
    ax_graph.autoscale_view()
    
    # Сохранение скриншотов на ключевых шагах
    if frame in [0, 50, 200, 1000]:
        filename = f"screenshot_step_{frame}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Сохранён скриншот: {filename}")
    
    return [cax, line_graph]

# 5. ЗАПУСК
ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, interval=50, blit=False)

plt.tight_layout()
plt.show()

print(f"Конец: Max = {rho.max().item():.4f}")
print(f"Скриншоты сохранены в: {os.getcwd()}")
