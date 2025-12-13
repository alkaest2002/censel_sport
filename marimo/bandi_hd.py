import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import pdfplumber
    import numpy as np
    import pandas as pd
    import marimo as mo

    output_path = Path("./bandi/")
    return Path, np, output_path, pd, pdfplumber


@app.cell
def _(pd, pdfplumber):
    def document_to_parse(document, pages):
        results = []
        with pdfplumber.open(document) as pdf:
            pages_to_extract = [pdf.pages[idx] for idx in pages ]
            for page in pages_to_extract:
                results.append(page.extract_tables())
        return results

    def convert_to_df(tables):
        df = pd.DataFrame()
        for t_gender, t_label, t_data in tables:
            t_df = pd.DataFrame(t_data[1:], columns=["awarded_score","value"])
            t_df = t_df.assign(test=t_label, gender=t_gender)
            df = pd.concat([df, t_df], ignore_index=True)
        df = df.assign(recruitment_type="hd", recrutiment_year=2021)
        return df.loc[:, ["recrutiment_year", "recruitment_type", "test", "gender", "value", "awarded_score"]]
    return convert_to_df, document_to_parse


@app.cell
def _(Path):
    #### HD 2021
    hd_2021 = Path("./bandi/hd/2021_hd.pdf")
    return (hd_2021,)


@app.cell
def _(document_to_parse, hd_2021, np):
    pages_2021 = document_to_parse(hd_2021, [9, 10])
    t_2021 = [np.asarray(t) for p in pages_2021 for t in p][:3]
    t_2021_0 = t_2021[0]
    t_2021_1 = np.r_[t_2021[1], t_2021[2]]
    return t_2021_0, t_2021_1


@app.cell
def _(np, t_2021_0, t_2021_1):
    tables_2021 = []
    tables_2021.append(("m", "1000mt_run", t_2021_0[1:9,:2]))
    tables_2021.append(("m","sit_ups", np.c_[t_2021_0[1:9, 0], t_2021_0[1:9, -1]]))
    tables_2021.append(("m","100mt_run", t_2021_0[9:,:2]))
    tables_2021.append(("m", "push_ups", np.c_[t_2021_0[9:, 0], t_2021_0[9:, -1]]))
    tables_2021.append(("f", "1000mt_run", t_2021_1[1:9,:2]))
    tables_2021.append(("f","sit_ups", np.c_[t_2021_1[1:9, 0], t_2021_1[1:9, -1]]))
    tables_2021.append(("f","100mt_run", t_2021_1[9:,:2]))
    tables_2021.append(("f", "push_ups", np.c_[t_2021_1[9:, 0], t_2021_1[9:, -1]]))
    return (tables_2021,)


@app.cell
def _(convert_to_df, output_path, tables_2021):
    df_2021 = convert_to_df(tables_2021)
    df_2021.to_csv(output_path / "2021_hd_tavole.csv", index=False)
    return


@app.cell
def _(Path):
    #### HD 2022
    hd_2022 = Path("./bandi/hd/2022_hd.pdf")
    return (hd_2022,)


@app.cell
def _(document_to_parse, hd_2022, np):
    pages_2022 = document_to_parse(hd_2022, [13])
    t_2022 = [np.asarray(t) for p in pages_2022 for t in p]
    t_2022_0 = t_2022[0]
    t_2022_1 =t_2022[1]
    return t_2022_0, t_2022_1


@app.cell
def _(np, t_2022_0, t_2022_1):
    tables_2022 = []
    tables_2022.append(("m", "1000mt_run", t_2022_0[1:9,:2]))
    tables_2022.append(("m","sit_ups", np.c_[t_2022_0[1:9, 0], t_2022_0[1:9, -1]]))
    tables_2022.append(("m","100mt_run", t_2022_0[9:,:2]))
    tables_2022.append(("m", "push_ups", np.c_[t_2022_0[9:, 0], t_2022_0[9:, -1]]))
    tables_2022.append(("f", "1000mt_run", t_2022_1[1:9,:2]))
    tables_2022.append(("f","sit_ups", np.c_[t_2022_1[1:9, 0], t_2022_1[1:9, -1]]))
    tables_2022.append(("f","100mt_run", t_2022_1[9:,:2]))
    tables_2022.append(("f", "push_ups", np.c_[t_2022_1[9:, 0], t_2022_1[9:, -1]]))
    return (tables_2022,)


@app.cell
def _(convert_to_df, output_path, tables_2022):
    df_2022 = convert_to_df(tables_2022)
    df_2022.to_csv(output_path / "2022_hd_tavole.csv", index=False)
    return


@app.cell
def _(Path):
    #### HD 2023
    hd_2023 = Path("./bandi/hd/2023_hd.pdf")
    return (hd_2023,)


@app.cell
def _(document_to_parse, hd_2023, np):
    pages_2023 = document_to_parse(hd_2023, [49])
    t_2023 = [np.asarray(t) for p in pages_2023 for t in p]
    t_2023_0 = t_2023[0]
    t_2023_1 =t_2023[1]
    return t_2023_0, t_2023_1


@app.cell
def _(np, t_2023_0, t_2023_1):
    tables_2023 = []
    tables_2023.append(("m", "1000mt_run", t_2023_0[1:9,:2]))
    tables_2023.append(("m","sit_ups", np.c_[t_2023_0[1:9, 0], t_2023_0[1:9, 2]]))
    tables_2023.append(("m","25mt_swim", np.c_[t_2023_0[1:9, 0], t_2023_0[1:9, -1]]))
    tables_2023.append(("m","100mt_run", t_2023_0[9:,:2]))
    tables_2023.append(("m", "push_ups", np.c_[t_2023_0[9:, 0], t_2023_0[9:, -1]]))
    tables_2023.append(("f", "1000mt_run", t_2023_1[1:9,:2]))
    tables_2023.append(("f","sit_ups", np.c_[t_2023_1[1:9, 0], t_2023_1[1:9, 2]]))
    tables_2023.append(("f","25mt_swim", np.c_[t_2023_1[1:9, 0], t_2023_1[1:9, -1]]))
    tables_2023.append(("f","100mt_run", t_2023_1[9:,:2]))
    tables_2023.append(("f", "push_ups", np.c_[t_2023_1[9:, 0], t_2023_1[9:, -1]]))
    return (tables_2023,)


@app.cell
def _(convert_to_df, output_path, tables_2023):
    df_2023 = convert_to_df(tables_2023)
    df_2023.to_csv(output_path / "2023_hd_tavole.csv", index=False)
    return


@app.cell
def _(Path):
    #### HD 2024
    hd_2024 = Path("./bandi/hd/2024_hd.pdf")
    return (hd_2024,)


@app.cell
def _(document_to_parse, hd_2024, np):
    pages_2024 = document_to_parse(hd_2024, [49, 50])
    t_2024 = [np.asarray(t) for p in pages_2024 for t in p][:-1]
    t_2024_0 = t_2024[0]
    t_2024_1 = np.r_[t_2024[1], np.c_[t_2024[2], None]]
    return pages_2024, t_2024_0, t_2024_1


@app.cell
def _(np, t_2024_0, t_2024_1):
    tables_2024 = []
    tables_2024.append(("m", "1000mt_run", t_2024_0[1:9,:2]))
    tables_2024.append(("m","sit_ups", np.c_[t_2024_0[1:9, 0], t_2024_0[1:9, 2]]))
    tables_2024.append(("m","25mt_swim", np.c_[t_2024_0[1:9, 0], t_2024_0[1:9, -1]]))
    tables_2024.append(("m","100mt_run", t_2024_0[9:,:2]))
    tables_2024.append(("m", "push_ups", np.c_[t_2024_0[9:, 0], t_2024_0[9:, -1]]))
    tables_2024.append(("f", "1000mt_run", t_2024_1[1:9,:2]))
    tables_2024.append(("f","sit_ups", np.c_[t_2024_1[1:9, 0], t_2024_1[1:9, 2]]))
    tables_2024.append(("f","25mt_swim", np.c_[t_2024_1[1:9, 0], t_2024_1[1:9, -1]]))
    tables_2024.append(("f","100mt_run", t_2024_1[9:,:2]))
    tables_2024.append(("f", "push_ups", np.c_[t_2024_1[9:, 0], t_2024_1[9:, -1]]))
    return (tables_2024,)


@app.cell
def _(convert_to_df, output_path, tables_2024):
    df_2024 = convert_to_df(tables_2024)
    df_2024.to_csv(output_path / "2024_hd_tavole.csv", index=False)
    return


@app.cell
def _(Path):
    #### HD 2025
    hd_2025 = Path("./bandi/hd/2025_hd.pdf")
    return


@app.cell
def _(document_to_parse, hd_2024, np, pages_2024):
    pages_2025 = document_to_parse(hd_2024, [50, 51])
    t_2025 = [np.asarray(t) for p in pages_2024 for t in p][:-1]
    t_2025_0 = t_2025[0]
    t_2025_1 = np.r_[t_2025[1], np.c_[t_2025[2], None]]
    return t_2025_0, t_2025_1


@app.cell
def _(np, t_2025_0, t_2025_1):
    tables_2025 = []
    tables_2025.append(("m", "1000mt_run", t_2025_0[1:9,:2]))
    tables_2025.append(("m","sit_ups", np.c_[t_2025_0[1:9, 0], t_2025_0[1:9, 2]]))
    tables_2025.append(("m","25mt_swim", np.c_[t_2025_0[1:9, 0], t_2025_0[1:9, -1]]))
    tables_2025.append(("m","100mt_run", t_2025_0[9:,:2]))
    tables_2025.append(("m", "push_ups", np.c_[t_2025_0[9:, 0], t_2025_0[9:, -1]]))
    tables_2025.append(("f", "1000mt_run", t_2025_1[1:9,:2]))
    tables_2025.append(("f","sit_ups", np.c_[t_2025_1[1:9, 0], t_2025_1[1:9, 2]]))
    tables_2025.append(("f","25mt_swim", np.c_[t_2025_1[1:9, 0], t_2025_1[1:9, -1]]))
    tables_2025.append(("f","100mt_run", t_2025_1[9:,:2]))
    tables_2025.append(("f", "push_ups", np.c_[t_2025_1[9:, 0], t_2025_1[9:, -1]]))
    return (tables_2025,)


@app.cell
def _(convert_to_df, output_path, tables_2025):
    df_2025 = convert_to_df(tables_2025)
    df_2025.to_csv(output_path / "2025_hd_tavole.csv", index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
