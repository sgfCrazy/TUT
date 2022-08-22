import pandas as pd

excel_abspath = r'C:\Users\86158\Desktop\TUT\TUT\Research\DL\test\0820终极.xlsx'
new_excel_abspath = r'C:\Users\86158\Desktop\TUT\TUT\Research\DL\test\0820终极_new.xlsx'

f = pd.ExcelFile(excel_abspath)

frame = pd.read_excel(excel_abspath, sheet_name=f.sheet_names)

print(f.sheet_names)

# frame.to_excel(new_excel_abspath)


writer = pd.ExcelWriter(new_excel_abspath)

for sheet_name, df in frame.items():
    print(f"写入： {sheet_name}")
    df.to_excel(writer, sheet_name=sheet_name)
    df.to_excel(writer, sheet_name=sheet_name)

writer.save()  # 储存文件
writer.close()  # 关闭writer

print()
