import re
from datetime import date
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, ConfigDict, field_validator
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, ForeignKey, Identity
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
import jwt
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/pizza_shop_2")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
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
    phone: str
    password: str
    is_admin: bool = False

    @field_validator('phone')
    def validate_phone(cls, v):
        v = v.strip()
        cleaned = v.replace(" ", "")
        if not v:
            raise ValueError('Phone number cannot be empty')
        if len(v) < 10:
            raise ValueError('Phone number is too short')
        if len(cleaned) > 12:
            raise ValueError('Phone number is too long')
        if not re.fullmatch(r'\+?\d+', cleaned):
            raise ValueError('Phone number must contain only digits and optional + at the beginning')
        return v

    @field_validator('password')
    def validate_password(cls, v):
        v = v.strip()
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if len(v) > 128:
            raise ValueError('Password must be at most 128 characters long')

        return v


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    password: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    phone: str
    is_admin: bool

    model_config = ConfigDict(from_attributes=True)


class ProductCreate(BaseModel):
    name: str
    description: str
    price: float
    is_available: bool = True
    image_url: str


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    is_available: Optional[bool] = None
    image_url: Optional[str] = None


class ProductResponse(BaseModel):
    id: int
    name: str
    description: str
    price: float
    is_available: bool
    image_url: str

    model_config = ConfigDict(from_attributes=True)


class CartItemCreate(BaseModel):
    product_id: int
    quantity: int


class CartItemUpdate(BaseModel):
    quantity: Optional[int] = None


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
    items: List[CartItemResponse] = []

    model_config = ConfigDict(from_attributes=True)


class OrderCreate(BaseModel):
    cart_id: int


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


def truncate_password(password: str, max_bytes: int = 72) -> str:
    """Укорачивает пароль до максимального количества байт"""
    encoded = password.encode('utf-8')
    if len(encoded) <= max_bytes:
        return password

    truncated = password
    while len(truncated.encode('utf-8')) > max_bytes:
        truncated = truncated[:-1]

    return truncated


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Верификация пароля с обработкой длинных паролей"""
    try:
        plain_password = truncate_password(plain_password)
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Хеширование пароля с обработкой длинных паролей"""
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


async def get_current_user(authorization: HTTPAuthorizationCredentials = Depends(security),
                           db: Session = Depends(get_db)):
    token_data = verify_token(authorization.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == token_data.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


async def get_current_admin(user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return user


@app.post("/login", response_model=LoginResponse, tags=["Authentication"])
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Аутентификация пользователя"""
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
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
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Создание нового пользователя"""
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        name=user.name,
        phone=user.phone,
        password=hashed_password,
        is_admin=user.is_admin
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return UserResponse.model_validate(db_user)


@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
def get_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Получение пользователя по ID"""
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return UserResponse.model_validate(user)


@app.put("/users/{user_id}", response_model=UserResponse, tags=["Users"])
def update_user(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    """Обновление пользователя"""
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    update_data = user_update.model_dump(exclude_unset=True)
    if "password" in update_data:
        update_data["password"] = get_password_hash(update_data["password"])

    for field, value in update_data.items():
        setattr(user, field, value)

    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@app.delete("/users/{user_id}", tags=["Users"])
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Удаление пользователя"""
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}


@app.post("/products", response_model=ProductResponse, tags=["Products"], dependencies=[Depends(get_current_admin)])
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    """Создание нового продукта (только для администратора)"""
    db_product = Product(**product.model_dump())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return ProductResponse.model_validate(db_product)


@app.get("/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def get_product(product_id: int, db: Session = Depends(get_db)):
    """Получение продукта по ID"""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    return ProductResponse.model_validate(product)


@app.get("/products", response_model=List[ProductResponse], tags=["Products"])
def get_all_products(db: Session = Depends(get_db)):
    """Получение всех продуктов"""
    products = db.query(Product).all()
    return [ProductResponse.model_validate(p) for p in products]


@app.put("/products/{product_id}", response_model=ProductResponse, tags=["Products"],
         dependencies=[Depends(get_current_admin)])
def update_product(product_id: int, product_update: ProductUpdate, db: Session = Depends(get_db)):
    """Обновление продукта (только для администратора)"""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )

    update_data = product_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(product, field, value)

    db.commit()
    db.refresh(product)
    return ProductResponse.model_validate(product)


@app.delete("/products/{product_id}", tags=["Products"], dependencies=[Depends(get_current_admin)])
def delete_product(product_id: int, db: Session = Depends(get_db)):
    """Удаление продукта (только для администратора)"""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )

    db.delete(product)
    db.commit()
    return {"message": "Product deleted successfully"}


@app.post("/cart/items", response_model=AddItemToCartResponse, tags=["Cart"])
def add_item_to_cart(
        item: CartItemCreate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Добавление товара в корзину"""
    product = db.query(Product).filter(Product.id == item.product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
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

    if existing_item:
        existing_item.quantity += item.quantity
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
    """Получение корзины пользователя"""
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()

    if not cart:
        return CartResponse(
            id=0,
            user_id=current_user.id,
            total_quantity=0,
            items=[]
        )

    total_quantity = sum(item.quantity for item in cart.items)

    cart_items_response = []
    for item in cart.items:
        item_data = {
            "product_id": item.product_id,
            "quantity": item.quantity
        }
        cart_items_response.append(CartItemResponse.model_validate(item_data))

    return CartResponse(
        id=cart.id,
        user_id=cart.user_id,
        total_quantity=total_quantity,
        items=cart_items_response
    )


@app.put("/cart/items/{product_id}", response_model=AddItemToCartResponse, tags=["Cart"])
def update_cart_item(
        product_id: int,
        item_update: CartItemUpdate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Обновление количества товара в корзине"""
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cart not found"
        )

    cart_item = db.query(CartItem).filter(
        CartItem.cart_id == cart.id,
        CartItem.product_id == product_id
    ).first()

    if not cart_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found in cart"
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
    """Удаление товара из корзины"""
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cart not found"
        )

    cart_item = db.query(CartItem).filter(
        CartItem.cart_id == cart.id,
        CartItem.product_id == product_id
    ).first()

    if not cart_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found in cart"
        )

    db.delete(cart_item)
    db.commit()
    return {"message": "Item removed from cart successfully"}


@app.delete("/cart", tags=["Cart"])
def clear_cart(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Очистка корзины пользователя"""
    cart = db.query(Cart).filter(Cart.user_id == current_user.id).first()
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cart not found"
        )

    db.query(CartItem).filter(CartItem.cart_id == cart.id).delete()
    db.commit()
    return {"message": "Cart cleared successfully"}


@app.post("/orders", response_model=OrderResponse, tags=["Orders"])
def create_order(
        order: OrderCreate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Создание нового заказа"""
    cart = db.query(Cart).filter(Cart.id == order.cart_id).first()
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cart not found"
        )

    if cart.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cart does not belong to the current user"
        )

    db_order = Order(cart_id=order.cart_id, user_id=current_user.id)
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    return OrderResponse.model_validate(db_order)


@app.get("/orders/{order_id}", response_model=OrderResponse, tags=["Orders"])
def get_order(
        order_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Получение заказа по ID"""
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )

    if order.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return OrderResponse.model_validate(order)


@app.get("/orders", response_model=List[OrderResponse], tags=["Orders"])
def get_user_orders(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Получение всех заказов пользователя"""
    orders = db.query(Order).filter(Order.user_id == current_user.id).all()
    return [OrderResponse.model_validate(o) for o in orders]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
