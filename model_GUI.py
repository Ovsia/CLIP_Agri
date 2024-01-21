import tkinter as tk
from tkinter import filedialog, OptionMenu
from tkinter import *
from PIL import ImageTk, Image
import torch
import clip
import wikipediaapi
from ttkbootstrap import Style

uploaded_image_path = None
# 导入训练好的模型
device = 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
checkpoint = torch.load("C:/Users/Lenovo/Desktop/CLIP_agri_gpu.pt")
model.load_state_dict(checkpoint)

# 初始化tkinter对象 并设置参数
top = tk.Tk()
top.geometry("800x600")  # 设置图形窗口的宽和高
top.title("Plants Classification Based on CLIP/马清源 21121531")  # 标题设置

top.configure(background="#CDCDCD")  # 背景色
# Label控件：指定的窗口top中显示的文本和图像
label = Label(top, background="#CDCDCD", font=("arial", 15, "bold"))

sign_image = Label(top)


def display_plant_info_from_wikipedia(classification):
    selected_language = language_var.get()
    if selected_language == '中文' :
        classes_chinese = {
            '水果': ['杏仁', '香蕉', '苹果', '橙子', '葡萄', '菠萝', '芒果', '草莓', '桃子'],
            '蔬菜': ['芦笋', '芥菜', '卷心菜', '胡萝卜', '番茄', '西兰花', '菠菜', '甜椒', '黄瓜', '萝卜'],
            '花朵': ['银莲花', '甘菊', '苜蓿草', '水仙花', '雏菊', '风信子花', '早樱花', '长寿花', '长寿花',
                     '迷你康乃馨紫色',
                     '菱形鳢尾草花', '一串红花', '小丽花', '蓝花', '大丽花', '紫花爬墙虎花', '仙人掌果实', '蔷薇',
                     '长寿花',
                     '鸽子草花', '瓦克斯花', '野葡萄藤'],
            '农田作物': ['小麦', '玉米', '大米', '大豆', '大麦', '棉花', '马铃薯', '甜菜根', '燕麦', '葵花']
        }

        classes_english = {
            '水果': ['Almond', 'Banana', 'Apple', 'Orange', 'Grapes', 'Pineapple', 'Mango', 'Strawberry', 'Peach'],
            '蔬菜': ['Asparagus', 'Mustard', 'Cabbage', 'Carrot', 'Tomato', 'Broccoli', 'Spinach', 'Bell Pepper',
                     'Cucumber', 'Radish'],
            '花朵': ['Anemone flower', 'Chamomile', 'Chickweed', 'Daffodil flower', 'Daisy Fleabane', 'Hyacinth flower',
                     'Jonquil flower', 'Lisianthus flower', 'Madagascar Periwinkle', 'Mini Carnation purple',
                     'Pickerelweed flower', 'Poinsettia flower', 'Pompon flower', 'Primrose blue', 'Protea',
                     'Purple Deadnettle flower', 'Ranunculus flower', 'Rose', 'Trachelium flower',
                     'Vervain Mallow flower', 'Waxflower', 'Wild Grape Vine'],
            '农田作物': ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Barley', 'Cotton', 'Potato', 'Sugar Beet', 'Oats',
                         'Sunflower']
        }

        class_mapping_en_to_cn = {}
        for category in classes_chinese:
            for i, english_name in enumerate(classes_english[category]):
                chinese_name = classes_chinese[category][i]
                class_mapping_en_to_cn[english_name] = chinese_name

        classification = class_mapping_en_to_cn.get(classification)
        print(classification)
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'zh' if selected_language == '中文' else 'en')
    page_py = wiki_wiki.page(classification)

    info_text = f"植物信息：{classification}\n\n"

    if page_py.exists():
        if selected_language == 'English':
            info_text += page_py.text[:400]  # 获取页面文本
        if selected_language == '中文':
            info_text += page_py.text[:300]
    else:
        info_text += "抱歉，未找到此植物的详细信息。"

    plant_info_label.config(text=info_text)


def classify(file_path, selected_category):
    global label_packed

    classes_by_category = {
        '水果': ['Almond', 'Banana', 'Apple', 'Orange', 'Grapes', 'Pineapple', 'Mango', 'Strawberry',
                 'Peach'],
        '蔬菜': ['Asparagus', 'Mustard', 'Cabbage', 'Carrot', 'Tomato', 'Broccoli', 'Spinach', 'Bell Pepper',
                 'Cucumber', 'Radish'],
        '花朵': ['Anemone flower', 'Chamomile', 'Chickweed', 'Daffodil flower', 'Daisy Fleabane', 'Hyacinth flower',
                 'Jonquil flower', 'Lisianthus flower', 'Madagascar Periwinkle', 'Mini Carnation purple',
                 'Pickerelweed flower', 'Poinsettia flower', 'Pompon flower', 'Primrose blue', 'Protea',
                 'Purple Deadnettle flower', 'Ranunculus flower', 'Rose', 'Trachelium flower',
                 'Vervain Mallow flower', 'Waxflower', 'Wild Grape Vine'],
        '农田作物': ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Barley', 'Cotton', 'Potato', 'Sugar Beet', 'Oats',
                     'Sunflower']
    }

    classes = classes_by_category.get(selected_category, [])

    if not classes:
        label.configure(foreground="#FF0000", text="无效的类别选择")
        return

    preprocessed_text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    preprocessed_image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_logits, text_logits = model(preprocessed_image, preprocessed_text)
    proba_list = image_logits.softmax(dim=-1).cpu()
    values, indices = proba_list[0].topk(1)
    sign = classes[indices.item()]
    label.configure(foreground="#011638", text=f"分类结果: {sign}")
    display_plant_info_from_wikipedia(sign)


def show_classify_button(file_path, selected_category):
    """
    功能：实现点击按钮
    """
    # 实例化按钮对象
    classify_b = Button(top,  # 位置
                        text="Classify Image",  # 按钮名称
                        command=lambda: classify(file_path, selected_category),  # 执行回调函数
                        padx=10,  # 指定水平和垂直方向上按钮内容和边框的间距
                        pady=5
                        )
    # 对象属性
    classify_b.configure(style="secondary.TButton",  # 使用ttkbootstrap样式
                         foreground='white',  # 前景色
                         font=('arial', 10, 'bold'))  # 调整字体
    # place: relx, rely代表窗口大小所对应的x, y坐标比例
    classify_b.place(relx=0.8, rely=0.5)


def on_category_change(*args):
    selected_category = category_var.get()
    # Do something based on the selected category, if needed


def upload_image():
    """
    图片加载功能
    """
    global uploaded_image_path
    try:
        file_path = filedialog.askopenfilename()  # 打开文件返回文件名；本地目录也是整体的路径
        uploaded_image_path = file_path
        uploaded = Image.open(file_path)  # 打开图片，PIL类型；pytorch的顺序：(batch,c,h,w)  tf和numpy是(batch,h,w,c)
        uploaded.thumbnail(((top.winfo_width() / 2.25),  # 对图片实施裁剪
                            (top.winfo_height() / 2.25)))

        im = ImageTk.PhotoImage(uploaded)  # tk.PhotoImage(file=path_to_image)
        sign_image.configure(image=im)  # sign_image = Label(top)  Label实例对象中配置im图片
        sign_image.image = im
        label.configure(text='')
    except Exception as e:
        print(e)

style = Style(theme="lumen")  # 选择主题样式，可以尝试不同的主题
style.theme_use()

# 标题（上面）
title = Label(top,  # 位置
              text="Plants Classification",  # 标题内容
              pady=20,  # 和y轴边距
              font=('arial', 25, 'bold'))  # 字体设置

title.configure(background='#41B3A3', foreground='#FFFFFF')
# 背景色和前景色
title.pack()  # 布局


# 创建Frame
frame = Frame(top, background="#CDCDCD")
frame.pack(side=TOP, fill="both", expand=True)

# 图片显示在Frame的左侧
sign_image = Label(frame)
sign_image.pack(side=LEFT, padx=10, pady=10)

# 文字显示在图片的右侧
plant_info_label = Label(frame, background="#CDCDCD", font=("arial", 12, "italic"), wraplength=400)
plant_info_label.pack(side=LEFT, padx=10, pady=10)

# 创建两个Frame，一个放在最底部左侧，一个放在最底部右侧
bottom_left_frame = Frame(top, background="#CDCDCD")
bottom_left_frame.pack(side=LEFT, fill="x")

bottom_right_frame = Frame(top, background="#CDCDCD")
bottom_right_frame.pack(side=RIGHT, fill="x")

# 上传图片按钮
upload_button = Button(bottom_left_frame,
                       text="Upload an image",
                       command=upload_image,
                       padx=10,
                       pady=5)
upload_button.pack(side=BOTTOM, padx=10, pady=10)

classify_button = Button(bottom_right_frame,
                         text="Classify Image",
                         command=lambda: classify(uploaded_image_path, category_var.get()),
                         padx=10,
                         pady=5)
classify_button.pack(side=BOTTOM, padx=10, pady=10)



categories = ['水果', '蔬菜', '花朵', '农田作物']
category_var = tk.StringVar()
category_var.set(categories[0])  # 默认选择第一个类别
category_menu = OptionMenu(top, category_var, *categories)
category_menu.pack(pady=20)
category_var.trace_add('write', on_category_change)  # 监听类别变化

languages = ['中文', 'English']
language_var = tk.StringVar()
language_var.set(languages[0])  # 默认选择中文
language_menu = OptionMenu(top, language_var, *languages)
language_menu.pack(pady=20)


top.mainloop()  # 运行图片对象 top = tk.Tk()

