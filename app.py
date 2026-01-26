import re
from datetime import date, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, ConfigDict, field_validator, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, ForeignKey, Identity
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
import jwt
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/pizza_shop_2")
SECRET_KEY = os.getenv("SECRET_KEY", "your-very-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="Online Shop API",
    description="API для онлайн-магазина",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

security = HTTPBearer()

pwd_context = CryptContext(
    schemes=["sha256_crypt", "bcrypt"],
    default="sha256_crypt",
    deprecated="auto"
)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, Identity(start=1, increment=1), primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)

    cart = relationship("Cart", back_populates="user", uselist=False)
    orders = relationship("Order", back_populates="user")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, Identity(start=1, increment=1), primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    is_available = Column(Boolean, default=True, nullable=False)
    image_url = Column(String, nullable=False)
    stock_quantity = Column(Integer, default=0, nullable=False)


class CartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, Identity(start=1, increment=1), primary_key=True, index=True)
    cart_id = Column(Integer, ForeignKey("carts.id", ondelete="CASCADE"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    quantity = Column(Integer, nullable=False)

    product = relationship("Product")
    cart = relationship("Cart", back_populates="items")


class Cart(Base):
    __tablename__ = "carts"

    id = Column(Integer, Identity(start=1, increment=1), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)

    user = relationship("User", back_populates="cart")
    items = relationship("CartItem", back_populates="cart", cascade="all, delete-orphan")
    order = relationship("Order", back_populates="cart", uselist=False)


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, Identity(start=1, increment=1), primary_key=True, index=True)
    cart_id = Column(Integer, ForeignKey("carts.id"), nullable=False)
    created_at = Column(Date, default=date.today, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    cart = relationship("Cart", back_populates="order")
    user = relationship("User", back_populates="orders")


def recreate_tables():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


recreate_tables()


class UserCreate(BaseModel):
    email: EmailStr
    name: str
    phone: str = Field(examples=['+71234567890'])
    password: str
    is_admin: bool = False

    @field_validator('phone')
    def validate_phone(cls, v):
        v = v.strip()
        cleaned = v.replace(" ", "")
        if not v:
            raise ValueError('Номер телефона не может быть пустым')
        if len(v) < 10:
            raise ValueError('Номер телефона должен содержать не менее 10 цифр')
        if len(cleaned) > 12:
            raise ValueError('Номер телефона должен содержать не более 12 цифр')
        if not re.fullmatch(r'\+?\d+', cleaned):
            raise ValueError('Номер телефона должен содержать только цифры и опционально символ + в начале')
        return v

    @field_validator('password')
    def validate_password(cls, v):
        v = v.strip()
        if len(v) < 6:
            raise ValueError('Пароль должен содержать не менее 6 символов')
        if len(v) > 128:
            raise ValueError('Пароль должен содержать не более 128 символов')

        return v


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    phone: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    phone: str
    is_admin: bool

    model_config = ConfigDict(from_attributes=True)


class UserRegister(BaseModel):
    email: EmailStr
    name: str
    phone: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100,
                      description="Название продукта минимум 2 символа")
    description: str = Field(..., min_length=10, max_length=1000,
                             description="Описание продукта минимум 10 символов")
    price: float = Field(..., gt=0, le=1000000,
                         description="Цена от 1 до 1,000,000")
    is_available: bool = True
    image_url: str = Field(..., description="URL изображения продукта", examples=['https://example.com/image.jpg'])
    stock_quantity: int = Field(..., ge=0, le=100000,
                                description="Количество товара в наличии от 0 до 1,000,000", examples=[10])

    @field_validator('name')
    def validate_name(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError('Название должно содержать минимум 2 символа')
        if not re.match(r'^[a-zA-Zа-яА-Я0-9\s\-_.,]+$', v):
            raise ValueError('Название содержит недопустимые символы')
        return v

    @field_validator('description')
    def validate_description(cls, v):
        v = v.strip()
        if len(v) < 10:
            raise ValueError('Описание должно содержать минимум 10 символов')
        if len(v) > 1000:
            raise ValueError('Описание не должно превышать 1000 символов')
        return v

    @field_validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Цена должна быть больше 0')
        if v > 1000000:
            raise ValueError('Цена не должна превышать 1,000,000')
        return round(v, 2)

    @field_validator('image_url')
    def validate_image_url(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('URL изображения не может быть пустым')

        url_pattern = re.compile(
            r'^https?://'  # http:// или https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # домен
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # или IP
            r'(?::\d+)?'  # порт
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if not url_pattern.match(v):
            raise ValueError('Некорректный URL изображения')

        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if not any(v.lower().endswith(ext) for ext in image_extensions):
            raise ValueError('URL должен быть в формате: jpg, jpeg, png, webp')

        return v

    @field_validator('stock_quantity')
    def validate_stock_quantity(cls, v):
        if v < 0:
            raise ValueError('Количество товара не может быть отрицательным')
        if v > 100000:
            raise ValueError('Количество товара не должно превышать 100,000')
        return v

    class Config:
        min_anystr_length = 1  # Минимальная длина для всех строковых полей
        validate_all = True  # Валидация всех полей


class ProductUpdate(BaseModel):
    """Схема для обновления продукта"""
    name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
        description="Название продукта минимум 2 символа"
    )
    description: Optional[str] = Field(
        default=None,
        min_length=10,
        max_length=1000,
        description="Описание продукта минимум 10 символов"
    )
    price: Optional[float] = Field(
        default=None,
        gt=0,
        le=1000000,
        description="Цена от 1 до 1,000,000"
    )
    is_available: Optional[bool] = None
    image_url: Optional[str] = Field(
        default=None,
        description="URL изображения продукта",
        examples=['https://example.com/image.jpg']
    )
    stock_quantity: Optional[int] = Field(
        default=None,
        ge=0,
        le=100000,
        description="Количество товара в наличии от 0 до 100,000",
        examples=[10]
    )


class ProductPutSchema(BaseModel):
    """Схема для ПОЛНОГО обновления (PUT) - все поля обязательны"""
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Название продукта минимум 2 символа"
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Описание продукта минимум 10 символов"
    )
    price: float = Field(
        ...,
        gt=0,
        le=1000000,
        description="Цена от 1 до 1,000,000"
    )
    is_available: bool = Field(
        ...,
        description="Доступен ли продукт для покупки"
    )
    image_url: str = Field(
        ...,
        description="URL изображения продукта",
        examples=['https://example.com/image.jpg']
    )
    stock_quantity: int = Field(
        ...,
        ge=0,
        le=100000,
        description="Количество товара в наличии от 0 до 100,000",
        examples=[10]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Смартфон Premium",
                "description": "Флагманский смартфон с OLED дисплеем, 512ГБ памяти и тройной камерой",
                "price": 89999.99,
                "is_available": True,
                "image_url": "https://example.com/images/smartphone.jpg",
                "stock_quantity": 50
            }
        }
    )


class ProductResponse(BaseModel):
    id: int
    name: str
    description: str
    price: float
    is_available: bool
    image_url: str
    stock_quantity: int

    model_config = ConfigDict(from_attributes=True)


class CartItemProductResponse(BaseModel):
    product_id: int
    quantity: int
    product_name: str
    product_price: float
    product_image_url: str
    is_available: bool
    has_enough_stock: bool
    available_quantity: int

    model_config = ConfigDict(from_attributes=True)


class CartItemCreate(BaseModel):
    product_id: int
    quantity: int = Field(..., gt=0, le=100, description="Количество должно быть от 1 до 100")

    @field_validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Количество должно быть положительным числом')
        if v > 100:
            raise ValueError('Нельзя добавить более 100 единиц товара за раз')
        return v


class CartItemUpdate(BaseModel):
    quantity: Optional[int] = Field(None, gt=0, le=100, description="Количество должно быть от 1 до 100")

    @field_validator('quantity')
    def validate_quantity(cls, v):
        if v is not None:
            if v <= 0:
                raise ValueError('Количество должно быть положительным числом')
            if v > 100:
                raise ValueError('Нельзя добавить более 100 единиц товара за раз')
        return v


class AddItemToCartResponse(BaseModel):
    product_id: int
    quantity: int
    cart_id: int

    model_config = ConfigDict(from_attributes=True)


class CartItemResponse(BaseModel):
    product_id: int
    quantity: int

    model_config = ConfigDict(from_attributes=True)


class CartResponse(BaseModel):
    id: int
    user_id: int
    total_quantity: int
    total_price: float
    items: List[CartItemProductResponse] = []

    model_config = ConfigDict(from_attributes=True)


class OrderCreate(BaseModel):
    cart_id: int

    @field_validator('cart_id')
    def validate_cart_id(cls, v):
        if v <= 0:
            raise ValueError('ID корзины должно быть положительным числом')
        return v


class OrderResponse(BaseModel):
    id: int
    cart_id: int
    created_at: date
    user_id: int

    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

    model_config = ConfigDict(from_attributes=True)


class TokenData(BaseModel):
    user_id: int
    is_admin: bool


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def validate_product_update(update_data: dict, partial: bool = False) -> list:
    """
    Валидация данных обновления продукта

    :param update_data: Словарь с данными для обновления
    :param partial: True для PATCH (частичное обновление)
    :return: Список ошибок валидации
    """
    errors = []

    # Валидация для поля name
    if 'name' in update_data and update_data['name'] is not None:
        name = update_data['name']
        if not partial and not name:  # Для PUT обязательно
            errors.append("Название продукта обязательно")
        elif name and (len(name) < 2 or len(name) > 100):
            errors.append("Название должно быть от 2 до 100 символов")

    # Валидация для поля description
    if 'description' in update_data and update_data['description'] is not None:
        desc = update_data['description']
        if not partial and not desc:  # Для PUT обязательно
            errors.append("Описание продукта обязательно")
        elif desc and (len(desc) < 10 or len(desc) > 1000):
            errors.append("Описание должно быть от 10 до 1000 символов")

    # Валидация для поля price
    if 'price' in update_data and update_data['price'] is not None:
        price = update_data['price']
        if not partial and price is None:  # Для PUT обязательно
            errors.append("Цена продукта обязательна")
        elif price is not None and (price <= 0 or price > 1000000):
            errors.append("Цена должна быть от 1 до 1,000,000")

    # Валидация для поля stock_quantity
    if 'stock_quantity' in update_data and update_data['stock_quantity'] is not None:
        stock = update_data['stock_quantity']
        if stock is not None and (stock < 0 or stock > 100000):
            errors.append("Количество должно быть от 0 до 100,000")

    # Валидация для поля image_url
    if 'image_url' in update_data and update_data['image_url'] is not None:
        if not partial and not update_data['image_url']:  # Для PUT обязательно
            errors.append("URL изображения обязателен")

    return errors


def truncate_password(password: str, max_bytes: int = 72) -> str:
    encoded = password.encode('utf-8')
    if len(encoded) <= max_bytes:
        return password

    truncated = password
    while len(truncated.encode('utf-8')) > max_bytes:
        truncated = truncated[:-1]

    return truncated


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        plain_password = truncate_password(plain_password)
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Ошибка при проверке пароля: {e}")
        return False


def get_password_hash(password: str) -> str:
    password = truncate_password(password)
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(user_id=payload.get("user_id"), is_admin=payload.get("is_admin", False))
    except jwt.PyJWTError:
        return None


@app.post("/register", response_model=TokenResponse, tags=["Authentication"])
def register(user: UserRegister, db: Session = Depends(get_db)):
    # Проверяем, существует ли уже пользователь с таким email
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email уже зарегистрирован"
        )

    # Хэшируем пароль
    hashed_password = get_password_hash(user.password)

    # Создаем пользователя (is_admin всегда False для регистрации)
    db_user = User(
        email=user.email,
        name=user.name,
        phone=user.phone,
        password=hashed_password,
        is_admin=False  # Всегда False при регистрации
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Создаем токен - передаем словарь с данными
    token_data = {
        "user_id": db_user.id,
        "email": db_user.email,
        "is_admin": db_user.is_admin
    }
    access_token = create_access_token(token_data)

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(db_user)
    )


async def get_current_user(authorization: HTTPAuthorizationCredentials = Depends(security),
                           db: Session = Depends(get_db)):
    token_data = verify_token(authorization.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидные данные авторизации",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == token_data.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )
    return user


async def get_current_admin(user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав"
        )
    return user


def check_product_availability(product_id: int, requested_quantity: int, db: Session) -> Product:
    product = db.query(Product).filter(
        Product.id == product_id,
        Product.is_available == True
    ).first()

    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Продукт не найден или недоступен"
        )

    # Проверка количества товара в наличии
    if requested_quantity > product.stock_quantity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Недостаточно товара в наличии"
        )

    return product


@app.post("/login", response_model=LoginResponse, tags=["Authentication"])
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный логин или пароль"
        )

    access_token = create_access_token({
        "user_id": user.id,
        "is_admin": user.is_admin
    })

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@app.post("/users", response_model=UserResponse, tags=["Users"])
def create_user(
        user: UserCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)  # Используем вашу функцию
):
    # Проверяем, что текущий пользователь - админ
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав"
        )

    # Проверяем, существует ли уже пользователь с таким email
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email уже зарегистрирован"
        )

    # Хэшируем пароль
    hashed_password = get_password_hash(user.password)

    # Создаем пользователя
    db_user = User(
        email=user.email,
        name=user.name,
        phone=user.phone,
        password=hashed_password,
        is_admin=user.is_admin  # Админ может установить is_admin из запроса
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return UserResponse.model_validate(db_user)


@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
def get_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )
    return UserResponse.model_validate(user)


@app.get("/user/me", response_model=UserResponse, tags=["Users"])
def get_current_user_info(
        current_user: User = Depends(get_current_user),
):

    return UserResponse.model_validate(current_user)


@app.put("/users/{user_id}", response_model=UserResponse, tags=["Users"])
def update_user(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )

    # Исключаем пароль из данных для обновления
    update_data = user_update.model_dump(exclude_unset=True, exclude={"password"})

    # Проверяем, что email остается уникальным (если обновляется email)
    if "email" in update_data:
        existing_user = db.query(User).filter(
            User.email == update_data["email"],
            User.id != user_id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Пользователь с таким email уже существует"
            )

    for field, value in update_data.items():
        setattr(user, field, value)

    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


# Схема для обновления пароля
class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str


@app.patch("/users/{user_id}/password", response_model=UserResponse, tags=["Users"])
def update_password(user_id: int, password_update: PasswordUpdate, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )

    # Проверка текущего пароля (для обычных пользователей)
    if not current_user.is_admin or current_user.id == user_id:
        if not verify_password(password_update.current_password, user.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Неверный текущий пароль"
            )

    # Обновление пароля
    user.password = get_password_hash(password_update.new_password)

    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@app.delete("/users/{user_id}", tags=["Users"])
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден"
        )

    db.delete(user)
    db.commit()
    return {"message": "Пользователь удален"}


@app.post("/products", response_model=ProductResponse, tags=["Products"], dependencies=[Depends(get_current_admin)])
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    db_product = Product(**product.model_dump())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return ProductResponse.model_validate(db_product)


@app.get("/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Продукт не найден"
        )
    return ProductResponse.model_validate(product)


@app.get("/products", response_model=List[ProductResponse], tags=["Products"])
def get_all_products(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    return [ProductResponse.model_validate(p) for p in products]


@app.put("/products/{product_id}",
         response_model=ProductResponse,
         tags=["Products"],
         summary="Полностью обновить продукт",
         description="Полное обновление продукта. ВСЕ поля обязательны, включая is_available и stock_quantity.",
         dependencies=[Depends(get_current_admin)])
def update_product(
        product_id: int,
        product_data: ProductPutSchema,  # Используем ProductPutSchema вместо ProductUpdate
        db: Session = Depends(get_db)
):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Продукт не найден"
        )

    # Преобразуем Pydantic модель в словарь
    update_data = product_data.model_dump()

    # Применяем все обновления
    for field, value in update_data.items():
        setattr(product, field, value)

    try:
        db.commit()
        db.refresh(product)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении продукта: {str(e)}"
        )

    return ProductResponse.model_validate(product)


@app.patch("/products/{product_id}",
           response_model=ProductResponse,
           tags=["Products"],
           summary="Частично обновить продукт",
           description="Обновляет только указанные поля продукта. Незаданные поля остаются без изменений.",
           dependencies=[Depends(get_current_admin)])
def partial_update_product(
        product_id: int,
        product_update: ProductUpdate,  # Используем ProductUpdate для PATCH
        db: Session = Depends(get_db)
):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Продукт не найден"
        )

    # Получаем только установленные (не None) поля
    update_data = product_update.model_dump(exclude_unset=True)

    # Если ничего не передано, возвращаем текущий продукт
    if not update_data:
        return ProductResponse.model_validate(product)

    # Валидация переданных полей
    errors = []

    if 'name' in update_data and update_data['name'] is not None:
        if len(update_data['name']) < 2 or len(update_data['name']) > 100:
            errors.append("Название должно быть от 2 до 100 символов")

    if 'description' in update_data and update_data['description'] is not None:
        if len(update_data['description']) < 10 or len(update_data['description']) > 1000:
            errors.append("Описание должно быть от 10 до 1000 символов")

    if 'price' in update_data and update_data['price'] is not None:
        if update_data['price'] <= 0 or update_data['price'] > 1000000:
            errors.append("Цена должна быть от 1 до 1,000,000")

    if 'stock_quantity' in update_data and update_data['stock_quantity'] is not None:
        if update_data['stock_quantity'] < 0 or update_data['stock_quantity'] > 100000:
            errors.append("Количество должно быть от 0 до 100,000")

    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": errors}
        )

    # Применяем обновления
    for field, value in update_data.items():
        setattr(product, field, value)

    try:
        db.commit()
        db.refresh(product)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении продукта: {str(e)}"
        )

    return ProductResponse.model_validate(product)


@app.delete("/products/{product_id}", tags=["Products"], dependencies=[Depends(get_current_admin)])
def delete_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Продукт не найден"
        )

    db.delete(product)
    db.commit()
    return {"message": "Продукт удален"}


@app.post("/cart/items", response_model=AddItemToCartResponse, tags=["Cart"])
def add_item_to_cart(
        item: CartItemCreate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    product = db.query(Product).filter(
        Product.id == item.product_id,
        Product.is_available == True
    ).first()

    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Продукт не найден или недоступен"
        )

    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()
    if not cart:
        cart = Cart(user_id=current_user.id)
        db.add(cart)
        db.commit()
        db.refresh(cart)

    existing_item = db.query(CartItem).filter(
        CartItem.cart_id == cart.id,
        CartItem.product_id == item.product_id
    ).first()

    new_total_quantity = item.quantity
    if existing_item:
        new_total_quantity += existing_item.quantity

    if new_total_quantity > product.stock_quantity:
        available = product.stock_quantity - (existing_item.quantity if existing_item else 0)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Недостаточно товара в наличии"
        )

    # Проверяем, что общее количество не превышает разумный лимит
    if new_total_quantity > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нельзя добавить более 100 единиц товара за раз"
        )

    if existing_item:
        existing_item.quantity = new_total_quantity
        db_item = existing_item
    else:
        db_item = CartItem(
            cart_id=cart.id,
            product_id=item.product_id,
            quantity=item.quantity
        )
        db.add(db_item)

    db.commit()
    db.refresh(db_item)

    return AddItemToCartResponse(
        product_id=item.product_id,
        quantity=db_item.quantity,
        cart_id=cart.id
    )


@app.get("/cart", response_model=CartResponse, tags=["Cart"])
def get_cart(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()

    if not cart:
        return CartResponse(
            id=0,
            user_id=current_user.id,
            total_quantity=0,
            total_price=0.0,
            items=[]
        )

    total_quantity = 0
    total_price = 0.0
    cart_items_response = []

    for item in cart.items:
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if product:
            item_total_price = product.price * item.quantity
            total_quantity += item.quantity
            total_price += item_total_price

            has_enough_stock = item.quantity <= product.stock_quantity

            item_data = {
                "product_id": item.product_id,
                "quantity": item.quantity,
                "product_name": product.name,
                "product_price": product.price,
                "product_image_url": product.image_url,
                "is_available": product.is_available,
                "has_enough_stock": has_enough_stock,
                "available_quantity": product.stock_quantity
            }
            cart_items_response.append(CartItemProductResponse.model_validate(item_data))

    return CartResponse(
        id=cart.id,
        user_id=cart.user_id,
        total_quantity=total_quantity,
        total_price=round(total_price, 2),
        items=cart_items_response
    )


@app.put("/cart/items/{product_id}", response_model=AddItemToCartResponse, tags=["Cart"])
def update_cart_item(
        product_id: int,
        item_update: CartItemUpdate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Корзина не найдена"
        )

    cart_item = db.query(CartItem).filter(
        CartItem.cart_id == cart.id,
        CartItem.product_id == product_id
    ).first()

    if not cart_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Товар не найден в корзине"
        )

    # Проверяем доступность продукта перед обновлением
    if item_update.quantity is not None and item_update.quantity > 0:
        product = db.query(Product).filter(
            Product.id == product_id,
            Product.is_available == True
        ).first()

        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Продукт не найден или недоступен"
            )

        # Проверяем, что новое количество не превышает доступное на складе
        if item_update.quantity > product.stock_quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Недостаточно товара в наличии"
            )

    update_data = item_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(cart_item, field, value)

    db.commit()
    db.refresh(cart_item)

    return AddItemToCartResponse(
        product_id=cart_item.product_id,
        quantity=cart_item.quantity,
        cart_id=cart.id
    )


@app.delete("/cart/items/{product_id}", tags=["Cart"])
def remove_item_from_cart(
        product_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Корзина не найдена"
        )

    cart_item = db.query(CartItem).filter(
        CartItem.cart_id == cart.id,
        CartItem.product_id == product_id
    ).first()

    if not cart_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Продукт не найден в корзине"
        )

    db.delete(cart_item)
    db.commit()
    return {"message": "Продукт удален из корзины"}


@app.delete("/cart", tags=["Cart"])
def clear_cart(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Корзина не найдена"
        )

    db.query(CartItem).filter(CartItem.cart_id == cart.id).delete()
    db.commit()
    return {"message": "Корзина очищена"}


@app.post("/orders", response_model=OrderResponse, tags=["Orders"])
def create_order(
        order: OrderCreate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    cart = db.query(Cart).filter(Cart.id == order.cart_id).first()
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Корзина не найдена"
        )

    if cart.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Корзина не принадлежит текущему пользователю"
        )

    # Проверяем, что корзина не пустая
    cart_items = db.query(CartItem).filter(CartItem.cart_id == cart.id).all()
    if not cart_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нельзя создать заказ с пустой корзиной"
        )

    # Проверяем доступность всех товаров в корзине И количество на складе
    for cart_item in cart_items:
        product = db.query(Product).filter(
            Product.id == cart_item.product_id,
            Product.is_available == True
        ).first()

        if not product:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="В корзине есть недоступные для заказа товары"
            )

        # Проверяем, что количество в корзине не превышает доступное на складе
        if cart_item.quantity > product.stock_quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Недостаточно товара в наличии"
            )

    # ВАЖНО: Уменьшаем количество товаров на складе после создания заказа
    for cart_item in cart_items:
        product = db.query(Product).filter(Product.id == cart_item.product_id).first()
        if product:
            product.stock_quantity -= cart_item.quantity
            if product.stock_quantity < 0:
                product.stock_quantity = 0

    db_order = Order(cart_id=order.cart_id, user_id=current_user.id)
    db.add(db_order)
    db.commit()
    db.refresh(db_order)

    return OrderResponse.model_validate(db_order)


@app.get("/cart/validate", tags=["Cart"])
def validate_cart(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()

    if not cart:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Корзина не найдена"
        )

    cart_items = db.query(CartItem).filter(CartItem.cart_id == cart.id).all()

    if not cart_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Корзина пустая"
        )

    validation_errors = []
    total_price = 0.0

    for cart_item in cart_items:
        product = db.query(Product).filter(Product.id == cart_item.product_id).first()

        if not product:
            validation_errors.append(f"Товар с ID {cart_item.product_id} не найден")
        elif not product.is_available:
            validation_errors.append(f"Товар '{product.name}' недоступен")
        elif cart_item.quantity > product.stock_quantity:
            validation_errors.append(
                "Недостаточно товара в наличии"
            )
        else:
            total_price += product.price * cart_item.quantity

    if validation_errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "errors": validation_errors,
                "message": "В корзине есть проблемы с товарами"
            }
        )

    return {
        "valid": True,
        "message": "Корзина готова к оформлению заказа",
        "total_items": len(cart_items),
        "total_price": round(total_price, 2),
        "can_checkout": True
    }


@app.get("/orders/{order_id}", response_model=OrderResponse, tags=["Orders"])
def get_order(
        order_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Заказ не найден"
        )

    if order.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав"
        )

    return OrderResponse.model_validate(order)


@app.get("/orders", response_model=List[OrderResponse], tags=["Orders"])
def get_user_orders(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    orders = db.query(Order).filter(Order.user_id == current_user.id).all()
    return [OrderResponse.model_validate(o) for o in orders]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
