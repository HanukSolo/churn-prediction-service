"""
Pydantic схемы для валидации входных данных API.
"""
from pydantic import BaseModel, Field


class CustomerData(BaseModel):

    tenure: int = Field(..., description="Количество месяцев с компанией", ge=0)
    MonthlyCharges: float = Field(..., description="Ежемесячная плата", ge=0.0)
    TotalCharges: float = Field(..., description="Общая сумма платежей", ge=0.0)
    Contract: str = Field(..., description="Тип контракта")
    InternetService: str = Field(..., description="Тип интернет-услуги")
    PaymentMethod: str = Field(..., description="Способ оплаты")
