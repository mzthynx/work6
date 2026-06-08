# CG Lab Work6: 质点-弹簧系统物理仿真

基于 **Taichi** 框架实现的布料模拟系统，对比三种数值积分方法（显式/半隐式/隐式欧拉）在质点-弹簧模型中的稳定性表现。

## 目录

- [实验目标](#实验目标)
- [实验原理](#实验原理)
  - [质点-弹簧模型](#21-质点-弹簧模型-mass-spring-model)
  - [数值积分方法](#22-数值积分方法-numerical-integration)
- [实验任务与实现](#实验任务与实现)
  - [任务1：场景初始化（GPU 同步）](#任务1场景初始化gpu-同步)
  - [任务2：力学计算与防爆处理](#任务2力学计算与防爆处理)
  - [任务3：积分求解器实现](#任务3积分求解器实现)
  - [任务4：渲染与 GGUI 交互](#任务4渲染与-ggui-交互)
- [运行方式](#运行方式)
- [效果展示](#效果展示)
- [架构设计要点](#架构设计要点)
- [三种积分方法的稳定性分析](#三种积分方法的稳定性分析)

---

## 实验目标

| 目标 | 说明 |
|------|------|
| 动态场景渲染 | 使用 Taichi 框架构建 3D 场景，学习 Taichi GGUI 编写交互面板 |
| 质点-弹簧模型 | 掌握基于物理的弹力与阻尼力计算方法，处理数值爆炸问题（速度钳制） |
| 数值积分方法对比 | 独立编写并比较显式欧拉、半隐式欧拉、隐式欧拉三种积分求解器的稳定性差异 |
| GPU 编程基础 | 学习 `ti.kernel` 与 `ti.func`，理解并行计算中的状态同步与 Kernel 启动开销优化 |

## 实验原理

### 2.1 质点-弹簧模型 (Mass-Spring Model)

质点-弹簧系统是计算机图形学中最经典的变形体模拟方法之一。本实验将布料离散化为 **20×20 网格状的质点集合**，质点之间通过**结构弹簧**相连。

**弹力公式**（Hooke's Law）：

$$f_a = -k_s(|x_a - x_b| - l) \frac{x_a - x_b}{|x_a - x_b|}$$

其中 $k_s$ 为弹簧劲度系数，$l$ 为弹簧原长，$x$ 为质点位置。

**阻尼力公式**（Damping force）：

$$f_d = -k_d v_a$$

引入阻尼力的目的是防止系统能量无限增加导致发散。

### 2.2 数值积分方法 (Numerical Integration)

根据牛顿第二定律 $a = F/m$，在离散时间步 $\Delta t$ 内通过数值积分解更新质点的速度 $v$ 和位置 $x$。

#### 显式欧拉 (Explicit Euler)

完全使用当前时刻的状态来预测下一时刻——**最简单但最不稳定**：

$$
\begin{align}
x_{t+1} &= x_t + v_t \Delta t \\
v_{t+1} &= v_t + a_t \Delta t
\end{align}
$$

#### 半隐式欧拉 (Semi-Implicit / Symplectic Euler)

先更新速度，然后用更新后的速度来更新位置——**能量稳定、游戏/图形学首选**：

$$
\begin{align}
v_{t+1} &= v_t + a_t \Delta t \\
x_{t+1} &= x_t + v_{t+1} \Delta t
\end{align}
$$

#### 隐式欧拉 (Implicit / Backward Euler)

使用未来时刻的状态来计算受力——**无条件稳定但需要迭代求解**：

$$
\begin{align}
v_{t+1} &= v_t + a_{t+1} \Delta t \\
x_{t+1} &= x_t + v_{t+1} \Delta t
\end{align}
$$

本实验使用**定点迭代法 (Fixed-point Iteration)** 近似求解隐式方程。

---

## 实验任务与实现

### 任务1：场景初始化（GPU 同步）

定义 20×20 布料网格，初始化质点的位置、速度、受力和弹簧拓扑结构。

**关键约束**：为保证多线程下 GPU 计算状态的同步，将初始化操作拆分为多个 `@ti.kernel`，在 Python 侧按顺序调用：

```python
@ti.kernel
def init_positions():
    """初始化质点位置与固定状态 — 固定第一排两个角点"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        # 固定左上角和右上角
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1

@ti.kernel
def init_springs():
    """初始化结构弹簧（右邻 + 下邻）"""
    for i, j in ti.ndrange(N, N):
        # 右侧结构弹簧
        if i < N - 1:
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
        # 下方结构弹簧
        if j < N - 1:
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])

def init_cloth():
    """Python 层顺序调用各 kernel，确保 GPU 同步"""
    num_springs[None] = 0   # 重置计数
    init_positions()
    init_springs()
    init_spring_indices()
```

### 任务2：力学计算与防爆处理

#### 力计算函数 (`@ti.func` 内联优化)

将受力计算声明为 `ti.func`，编译时强制内联到调用它的 kernel 中，有效减少 GPU 函数调用开销：

```python
@ti.func
def compute_forces_on(pos, vel, force):
    # 第一阶段：重力 + 阻尼（每个质点独立，无冲突）
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    # 第二阶段：弹簧力累加（atomic_add 保证线程安全）
    for i in range(num_springs[None]):
        # Hooke's Law 弹力计算 ...
        ti.atomic_add(force[idx_a], f_spring)
        ti.atomic_add(force[idx_b], -f_spring)
```

> **为什么用 `ti.atomic_add`？** 多个弹簧可能连接到同一质点，并行写入会产生数据竞争。`atomic_add` 保证累加操作的原子性。

#### 速度钳制函数

限制质点最大速度，防止显式欧拉等不稳定方法出现数值爆炸：

```python
@ti.func
def clamp_velocity(vel, idx):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity
```

### 任务3：积分求解器实现

为极致性能，将受力计算和位置/速度更新合并在**同一个 kernel** 中完成，最小化每帧循环内的 kernel 启动次数。

#### 显式欧拉 —— 用旧速度更新位置

```python
@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt          # 旧速度 → 位置
            v[i] += (f[i]/mass) * dt    # 旧力 → 速度
            clamp_velocity(v, i)
```

#### 半隐式欧拉 —— 先速度后位置（推荐）

```python
@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i]/mass) * dt    # 先更新速度
            clamp_velocity(v, i)
            x[i] += v[i] * dt           # 新速度 → 位置
```

#### 隐式欧拉 —— 定点迭代近似

使用 `ti.static` 在编译期展开迭代循环（零运行时开销），通过 3 轮定点迭代逼近隐式解：

```python
@ti.kernel
def step_implicit_iter():
    # 复制当前状态到预测场
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    # 定点迭代（ti.static 编译期展开）
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i]/mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    # 写回收敛后的状态
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]
```

### 任务4：渲染与 GGUI 交互

使用 `ti.ui.Window` 构建 3D 场景，通过 `window.GUI` 编写控制面板：

| 功能 | 实现 |
|------|------|
| 积分方法切换 | 三个 Button 按钮（显式/半隐式/隐式），切换时自动重置布料 |
| 暂停/恢复 | Pause / Resume Simulation 按钮 |
| 重置布料 | Reset Cloth 按钮 |
| 3D 视角控制 | 鼠标右键拖拽旋转（`track_user_inputs`） |
| 渲染内容 | 蓝色粒子（顶点）+ 白色线框（弹簧） |

```python
window = ti.ui.Window("Games101 - Mass Spring System", (800, 800))
# ... GUI 控制面板 ...
scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))   # 顶点
scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))  # 弹簧
```

每帧执行 **40 个物理子步**（40 × 5e-4s = 0.02s/帧 ≈ 实时速度）。

---

## 运行方式

### 环境依赖

```bash
pip install taichi
```

### 启动仿真

```bash
python main.py
```

启动后窗口左侧显示控制面板，可实时切换积分方法、暂停仿真或重置布料。

### 参数一览

| 参数 | 值 | 含义 |
|------|-----|------|
| `N` | 20 | 布料网格分辨率 (20×20 = 400 质点) |
| `mass` | 1.0 | 单个质点质量 (kg) |
| `dt` | 5e-4 | 时间步长 (秒) |
| `k_s` | 10000.0 | 弹簧劲度系数 (N/m) |
| `k_d` | 1.0 | 阻尼系数 |
| `max_velocity` | 50.0 | 速度钳制阈值 (m/s) |
| substeps/frame | 40 | 每帧物理子步数 |

---

## 效果展示

### 三种积分方法的直观对比

| 方法 | 稳定性 | 能量行为 | 适用场景 |
|------|--------|----------|----------|
| 显式欧拉 | 差（易爆炸） | 能量正偏（持续增加） | 仅用于教学对比 |
| 半隐式欧拉 | 好（条件稳定） | 能量微衰减 | 游戏物理、实时仿真 |
| 隐式欧拉 | 极好（无条件稳定） | 能量强衰减（偏"粘稠"） | 离线仿真、高刚度系统 |

> 在本实验参数设置下（$k_s=10000$, $\Delta t=5\times 10^{-4}$）：
> - **显式欧拉**会迅速产生数值爆炸，即使有速度钳制也难以维持稳定形态
> - **半隐式欧拉**能保持稳定的布料摆动，呈现自然的物理行为
> - **隐式欧拉**最为稳定，但因能量衰减较快，布料运动显得偏"粘滞"

---

## 架构设计要点

### GPU Kernel 拆分策略

```
┌─────────────────────────────────────────────────┐
│                  Python 主循环                    │
│                                                   │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ init_cloth() │  │   step_xxx() (×40次/帧)  │  │
│  │              │  │                          │  │
│  │ ┌──────────┐ │  │  ┌──────────────────────┐│  │
│  │ │init_pos  │ │  │  │ compute_forces_on()  ││  │
│  │ │(kernel)  │ │  │  │  (ti.func 内联)      ││  │
│  │ ├──────────┤ │  │  ├──────────────────────┤│  │
│  │ │init_spring│ │  │  │ 位置/速度更新        ││  │
│  │ │(kernel)  │ │  │  │ clamp_velocity()     ││  │
│  │ ├──────────┤ │  │  └──────────────────────┘│  │
│  │ │init_idx  │ │  │          ↓ GPU Sync       │  │
│  │ │(kernel)  │ │  └──────────────────────────┘  │
│  │ └──────────┘ │                                │
│  └──────┬───────┘                                │
│         ↓ Python 顺序调用保证同步                 │
└─────────────────────────────────────────────────┘
```

### 性能优化手段总结

| 优化手段 | 说明 |
|----------|------|
| **`ti.func` 内联** | 力计算和速度钳制声明为 `ti.func`，编译时内联到 kernel，消除函数调用开销 |
| **单 kernel 合并** | 受力计算 + 积分更新合并为一个 `@ti.kernel`，减少 CPU-GPU 数据传输 |
| **`ti.static` 循环展开** | 隐式欧拉的定点迭代在编译期展开，无运行时分支开销 |
| **`ti.atomic_add`** | 弹簧力累加使用原子操作，避免多线程写冲突 |
| **Kernel 拆分初始化** | 初始化拆分为多个 kernel + Python 顺序调用，保证 GPU 状态同步 |

---

## 三种积分方法的稳定性分析

### 核心区别

```
显式 Euler:    x(t) ──[v(t)]──→ x(t+1)     ← 用"旧的"速度推进
               v(t) ──[a(t)]──→ v(t+1)     ← 用"旧的"加速度推进

半隐式 Euler:  v(t) ──[a(t)]──→ v(t+1)     ← 先用"旧的"加速度更新速度
               x(t) ──[v(t+1)]→ x(t+1)     ← 再用"新的"速度推进位置 ✓

隐式 Euler:    v(t) ──[a(t+1)]→ v(t+1)    ← 用"未来的"加速度（需迭代求解）
               x(t) ──[v(t+1)]→ x(t+1)
```

**半隐式欧拉为何更稳定？**

半隐式欧拉是一种**辛 (Symplectic) 积分器**，它近似保恒系统的哈密顿量（能量），不会像显式欧拉那样持续向系统注入虚假能量。这使得它在同等时间步长下具有远超显式欧拉的稳定性边界。

**隐式欧拉的特点**

隐式欧拉是无条件稳定的——无论时间步长多大都不会爆炸。但它是一种**强耗散**方法，会持续从系统中移除能量，导致运动看起来"粘稠"。本实验通过仅做 3 次定点迭代来平衡精度与性能。

---

## 文件说明

```
work6/
├── main.py          # 主程序：完整实现四个任务的 Taichi 物理仿真代码
└── README.md        # 本报告
```
