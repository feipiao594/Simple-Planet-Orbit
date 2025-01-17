import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.widgets import Slider

def calculate_orbit_type(M, m, x0, y0, vx0, vy0):
    G = 6.67430e-11  # 引力常数
    r0 = np.sqrt(x0**2 + y0**2)
    l = m * (x0 * vy0 - y0 * vx0)
    c = (G * (M + m) * m ** 2) / (l**2)
    theta0 = np.arctan(
        -((1 / r0 - c) * vx0 + c * (vy0 * x0 * y0 / r0**2 + vx0 * x0**2 / r0**2)) /
        ((1 / r0 - c) * vy0 + c * (vx0 * x0 * y0 / r0**2 + vy0 * y0**2 / r0**2))
    )
    epsilon = (1 - c * r0) / (x0 * np.cos(theta0) + y0 * np.sin(theta0))

    if abs(epsilon) < abs(c):
        return f"elliptical\n r = 1/({c} \n+ ({epsilon}) \n* cos(theta - {theta0}))"
    elif abs(epsilon) == abs(c):
        return f"parabola\n r = 1/({c} \n+ ({epsilon}) \n* cos(theta - {theta0}))"
    elif abs(epsilon) > abs(c):
        return f"hyperbola\n r = 1/({c} \n+ ({epsilon}) \n* cos(theta - {theta0}))"

# 计算轨迹函数
def calculate_orbit(M, m, x0, y0, vx0, vy0, num_points):
    G = 6.67430e-11  # 引力常数
    r0 = np.sqrt(x0**2 + y0**2)
    l = m * (x0 * vy0 - y0 * vx0)
    c = (G * (M + m) * m ** 2) / (l**2)
    theta0 = np.arctan(
        -((1 / r0 - c) * vx0 + c * (vy0 * x0 * y0 / r0**2 + vx0 * x0**2 / r0**2)) /
        ((1 / r0 - c) * vy0 + c * (vx0 * x0 * y0 / r0**2 + vy0 * y0**2 / r0**2))
    )
    epsilon = (1 - c * r0) / (x0 * np.cos(theta0) + y0 * np.sin(theta0))

    theta = np.linspace(0, 2 * np.pi, num_points)
    r = 1 / (c + epsilon * np.cos(theta - theta0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def draw_velocity_vector(ax, x_pos, y_pos, vx, vy):
    # 获取当前坐标系范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 坐标范围宽度和高度
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    scale_factor = 0.05  # 设置速度向量占当前坐标范围的 10%

    # 归一化速度向量长度
    vx_normalized = vx * scale_factor * min(x_range, y_range) / np.sqrt(vx**2 + vy**2)
    vy_normalized = vy * scale_factor * min(x_range, y_range) / np.sqrt(vx**2 + vy**2)

    # 绘制速度向量
    arrow = FancyArrow(x_pos, y_pos, vx_normalized, vy_normalized,
                       color='orange', width=0.005 * min(x_range, y_range))
    ax.add_patch(arrow)

# 更新函数
def update(val):
    global arrow, direction_point, direction_circle, vx0, vy0
    M = slider_M.val
    m = slider_m.val
    x0 = slider_x0.val
    y0 = slider_y0.val
    speed = slider_speed.val

    k = speed / np.sqrt(vx0**2 + vy0**2)
    vx0 = k * vx0
    vy0 = k * vy0

    if dragging_small_mass:
        x0, y0 = small_mass.get_offsets()[0]

    # 更新 vx 和 vy 根据方向点和速度模
    if dragging_direction_point:
        direction_x, direction_y = direction_point.get_offsets()[0]
        dx, dy = direction_x - x0, direction_y - y0
        norm = np.sqrt(dx**2 + dy**2)
        vx0 = speed * dx / norm
        vy0 = speed * dy / norm

    # 重新计算轨迹
    x, y = calculate_orbit(M, m, x0, y0, vx0, vy0, int(slider_theta.val))
    line.set_data(x, y)

    # 删除旧的速度向量并绘制新的
    for patch in reversed(ax.patches):
        if isinstance(patch, FancyArrow):
            patch.remove()
    arrow = draw_velocity_vector(ax, x0, y0, vx0, vy0)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 坐标范围宽度和高度
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    scale_factor = 0.05  # 设置速度向量占当前坐标范围的 10%

    # 更新方向点的位置（固定在两倍方向向量位置）
    direction_x = x0 + 2 * vx0 / speed * scale_factor * min(x_range, y_range)
    direction_y = y0 + 2 * vy0 / speed * scale_factor * min(x_range, y_range)
    direction_point.set_offsets([direction_x, direction_y])

    # 更新紫色方向点的圆形轨迹
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = x0 + 2 * scale_factor * min(x_range, y_range) * np.cos(theta)
    circle_y = y0 + 2 * scale_factor * min(x_range, y_range) * np.sin(theta)
    direction_circle.set_data(circle_x, circle_y)

    # 更新小质量天体的位置
    small_mass.set_offsets([x0, y0])

    # 更新显示的数据
    direction_angle = np.arctan2(vy0, vx0) * 180 / np.pi  # 转换为角度
    info_text.set_text(
        f"Speed: {speed:.2e}\n"
        f"Direction Angle: {direction_angle:.2f}°\n"
        f"Position: ({x0:.2f}, {y0:.2f})\n"
        f"Velocity: ({vx0:.2e}, {vy0:.2e})\n"
        f"Type: {calculate_orbit_type(M, m, x0, y0, vx0, vy0)}"
    )

    # 更新画布
    fig.canvas.draw_idle()


# 鼠标拖动事件
def on_drag(event):
    global dragging_small_mass, dragging_direction_point
     # 检查事件数据是否有效
    if event.xdata is None or event.ydata is None:
        return  # 无效坐标，忽略事件
    if dragging_small_mass:
        # 获取当前坐标系的范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 限制坐标在范围内
        new_x = np.clip(event.xdata, xlim[0], xlim[1])
        new_y = np.clip(event.ydata, ylim[0], ylim[1])
        small_mass.set_offsets([new_x, new_y])
        slider_x0.set_val(new_x)
        slider_y0.set_val(new_y)
        update(None)
    elif dragging_direction_point:
        # 获取当前坐标系的范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 限制坐标在范围内
        new_x = np.clip(event.xdata, xlim[0], xlim[1])
        new_y = np.clip(event.ydata, ylim[0], ylim[1])
        direction_point.set_offsets([new_x, new_y])
        update(None)

def on_press(event):
    global dragging_small_mass, dragging_direction_point
    if small_mass.contains(event)[0]:
        dragging_small_mass = True
    elif direction_point.contains(event)[0]:
        dragging_direction_point = True
    else:
        dragging_small_mass = False
        dragging_direction_point = False

# 鼠标释放事件
def on_release(event):
    global dragging_small_mass, dragging_direction_point
    dragging_small_mass = False
    dragging_direction_point = False

# 鼠标滚轮事件支持放大缩小
def on_scroll(event):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = (xlim[1] - xlim[0]) * 0.1  # 缩放10%
    y_range = (ylim[1] - ylim[0]) * 0.1

    if event.button == 'up':  # 放大
        ax.set_xlim(xlim[0] + x_range, xlim[1] - x_range)
        ax.set_ylim(ylim[0] + y_range, ylim[1] - y_range)
    elif event.button == 'down':  # 缩小
        ax.set_xlim(xlim[0] - x_range, xlim[1] + x_range)
        ax.set_ylim(ylim[0] - y_range, ylim[1] + y_range)

    update(None)

# 初始参数
M = 5.972e24  # 大质量天体质量
m = 1e20  # 小质量天体质量
x0, y0 = 40.0, 60.0  # 初始位置
vx0, vy0 = -1e5, 2e6  # 初始速度
speed = np.sqrt(vx0**2 + vy0**2)  # 速度模
num_points = 1000  # 模拟点数

# 计算轨迹
x, y = calculate_orbit(M, m, x0, y0, vx0, vy0, num_points)

# 绘制轨迹和动态标记
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.4)

# 绘制轨迹
line, = ax.plot(x, y, label='Orbit', color='red')

# 在图形右上角添加文本显示区域
info_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right')

# 显示数据
direction_angle = np.arctan2(vy0, vx0) * 180 / np.pi  # 转换为角度
info_text.set_text(
        f"Speed: {speed:.2e}\n"
        f"Direction Angle: {direction_angle:.2f}°\n"
        f"Position: ({x0:.2f}, {y0:.2f})\n"
        f"Velocity: ({vx0:.2e}, {vy0:.2e})\n"
        f"Type: {calculate_orbit_type(M, m, x0, y0, vx0, vy0)}"
    )

dragging_small_mass = False
dragging_direction_point = False

# 绘制大质量天体位置
ax.scatter([0], [0], color='blue', label='Big Mass', s=100)

# 绘制小质量天体的当前位置（初始点）
small_mass = ax.scatter([x0], [y0], color='green', label='Small Mass', s=50)

# 获取当前坐标系范围
xlim = ax.get_xlim()
ylim = ax.get_ylim()

x_range = xlim[1] - xlim[0]
y_range = ylim[1] - ylim[0]
magic_number_1 = 12.35248447110898 / 14.582552722258564
magic_number_2 = 0.7291276361129281 / 0.4303807148975094
# 不明白为什么这里的 x_range 会突变，起始状态必须乘上两个值

scale_factor = 0.1 * magic_number_1# 设置速度向量占当前坐标范围的 10%

# 绘制方向点
direction_point = ax.scatter([x0 + 2 * vx0 / speed * scale_factor * min(x_range, y_range)], 
                             [y0 + 2 * vy0 / speed * scale_factor * min(x_range, y_range)],
                              color='purple', label='Direction Point', s=50)

theta = np.linspace(0, 2 * np.pi, 100)
circle_x = x0 + 2 * scale_factor * min(x_range, y_range) * np.cos(theta)
circle_y = y0 + 2 * scale_factor * min(x_range, y_range) * np.sin(theta)
direction_circle, = ax.plot(circle_x, circle_y, linestyle='--', color='purple', label='Direction Circle')

# 绘制速度向量
vx_normalized = vx0 * scale_factor * min(x_range, y_range) / np.sqrt(vx0**2 + vy0**2)
vy_normalized = vy0 * scale_factor * min(x_range, y_range) / np.sqrt(vx0**2 + vy0**2)

arrow = FancyArrow(x0, y0, vx_normalized, vy_normalized,
                   color='orange', width=0.005 * magic_number_2 * min(x_range, y_range))
ax.add_patch(arrow)

# 设置图例、标题和网格
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Orbit of a Small Mass around a Large Mass')
ax.grid(True)
ax.legend()
ax.axis('equal')  # 保持比例

# 添加滑块
ax_theta = plt.axes([0.18, 0.3, 0.65, 0.03])
ax_M = plt.axes([0.18, 0.25, 0.65, 0.03])
ax_m = plt.axes([0.18, 0.2, 0.65, 0.03])
ax_x0 = plt.axes([0.18, 0.15, 0.65, 0.03])
ax_y0 = plt.axes([0.18, 0.1, 0.65, 0.03])
ax_speed = plt.axes([0.18, 0.05, 0.65, 0.03])

slider_theta = Slider(ax_theta, 'Theta', 100, 1000000, valinit=num_points, valstep=100)
slider_M = Slider(ax_M, 'M', 1e20, 1e26, valinit=M)
slider_m = Slider(ax_m, 'm', 1e14, 1e24, valinit=m)
slider_x0 = Slider(ax_x0, 'x0', -500, 500, valinit=x0)
slider_y0 = Slider(ax_y0, 'y0', -500, 500, valinit=y0)
slider_speed = Slider(ax_speed, 'Speed', 1e3, 1e7, valinit=speed)

# 绑定滑块更新
slider_theta.on_changed(update)
slider_M.on_changed(update)
slider_m.on_changed(update)
slider_x0.on_changed(update)
slider_y0.on_changed(update)
slider_speed.on_changed(update)

# 绑定鼠标事件
dragging = False
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_drag)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('scroll_event', on_scroll)

plt.show()