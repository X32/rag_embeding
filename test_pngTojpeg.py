from PIL import Image
import os


def png_to_jpg_batch(input_folder, output_folder, background_color=(255, 255, 255), recursive=False):
    """
    批量转换文件夹内的 PNG 为 JPG
    :param input_folder: 输入文件夹路径（存放 PNG 图片）
    :param output_folder: 输出文件夹路径（保存 JPG 图片）
    :param background_color: 透明背景填充色
    :param recursive: 是否递归处理子文件夹，默认 False
    """
    # 确保输出文件夹存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹
    for root, dirs, files in os.walk(input_folder):
        # 若不递归，只处理当前文件夹（跳过子文件夹）
        if not recursive and root != input_folder:
            continue

        # 遍历所有文件，筛选 PNG 格式
        for file in files:
            if file.lower().endswith(".png"):
                # 构建输入路径和输出路径
                input_path = os.path.join(root, file)
                # 保持原文件名，将后缀改为 .jpg
                jpg_filename = os.path.splitext(file)[0] + ".jpg"
                # 保持原文件夹结构（若递归）
                relative_path = os.path.relpath(root, input_folder)
                output_root = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_root):
                    os.makedirs(output_root)
                output_path = os.path.join(output_root, jpg_filename)

                # 调用单张转换函数
                try:
                    png_image = Image.open(input_path).convert("RGBA")
                    jpg_background = Image.new("RGB", png_image.size, background_color)
                    jpg_background.paste(png_image, mask=png_image.split()[3])
                    jpg_background.save(output_path, "JPEG", quality=85)
                    print(f"成功：{input_path} -> {output_path}")
                except Exception as e:
                    print(f"失败：{input_path} -> {str(e)}")


# 示例调用
if __name__ == "__main__":
    input_dir = "/Users/zhaoyang/Desktop/img/src"  # 存放 PNG 的文件夹
    output_dir = "/Users/zhaoyang/Desktop/img/output"  # 保存 JPG 的文件夹
    png_to_jpg_batch(input_dir, output_dir, recursive=True)  # 递归处理子文件夹