from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Product:
    name: str
    link: str
    supplier_links: list[str] = field(default_factory=list)


def load_products(path: str) -> list[Product]:
    df = pd.read_excel(path, header=None, engine="openpyxl")
    products = []
    for _, row in df.iterrows():
        name = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if not name:
            continue
        link = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        supplier_links = []
        for col in [2, 3, 4]:
            val = row.iloc[col] if col < len(row) and pd.notna(row.iloc[col]) else None
            if val and str(val).strip().startswith("http"):
                supplier_links.append(str(val).strip())
        products.append(Product(name=name, link=link, supplier_links=supplier_links))
    return products
