from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4

import jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from grade_system.config import Settings, load_settings
from grade_system.persistence.models import RefreshToken, User

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AuthenticationError(ValueError):
    pass


class ConflictError(ValueError):
    pass


@dataclass
class IssuedTokens:
    access_token: str
    refresh_token: str
    expires_in: int


class AuthService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        if not self.settings.jwt_secret_key:
            raise ValueError("未检测到 JWT_SECRET_KEY，请先在系统环境变量中配置它。")

    def register_user(
        self,
        session: Session,
        *,
        username: str,
        password: str,
        display_name: str,
    ) -> User:
        normalized_username = username.strip()
        if not normalized_username:
            raise ValueError("用户名不能为空。")
        if len(password) < 8:
            raise ValueError("密码长度至少为 8 位。")

        existing_user = session.scalar(
            select(User).where(User.username == normalized_username)
        )
        if existing_user is not None:
            raise ConflictError("用户名已存在。")

        resolved_display_name = display_name.strip() or normalized_username
        user = User(
            username=normalized_username,
            password_hash=self.hash_password(password),
            display_name=resolved_display_name,
        )
        session.add(user)
        session.flush()
        return user

    def authenticate_user(
        self,
        session: Session,
        *,
        username: str,
        password: str,
    ) -> User:
        normalized_username = username.strip()
        user = session.scalar(select(User).where(User.username == normalized_username))
        if user is None or user.status != 1:
            raise AuthenticationError("用户名或密码错误。")
        if not self.verify_password(password, user.password_hash):
            raise AuthenticationError("用户名或密码错误。")

        user.last_login_at = utcnow()
        session.flush()
        return user

    def issue_tokens(self, session: Session, user: User) -> IssuedTokens:
        access_token = self._build_access_token(user)
        refresh_token = self._build_refresh_token(session, user)
        return IssuedTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.settings.jwt_access_token_expire_minutes * 60,
        )

    def refresh_access_token(self, session: Session, refresh_token: str) -> str:
        payload = self._decode_token(refresh_token, expected_type="refresh")
        token_jti = str(payload.get("jti", ""))
        token_record = session.scalar(
            select(RefreshToken).where(RefreshToken.token_jti == token_jti)
        )
        if token_record is None:
            raise AuthenticationError("Refresh token 无效。")
        if token_record.revoked_at is not None:
            raise AuthenticationError("Refresh token 已失效。")
        if token_record.expires_at < utcnow():
            raise AuthenticationError("Refresh token 已过期。")
        if token_record.token_hash != self._hash_token(refresh_token):
            raise AuthenticationError("Refresh token 校验失败。")

        user = session.get(User, token_record.user_id)
        if user is None or user.status != 1:
            raise AuthenticationError("用户不存在或已被禁用。")

        return self._build_access_token(user)

    def revoke_refresh_token(self, session: Session, refresh_token: str) -> None:
        payload = self._decode_token(refresh_token, expected_type="refresh")
        token_jti = str(payload.get("jti", ""))
        token_record = session.scalar(
            select(RefreshToken).where(RefreshToken.token_jti == token_jti)
        )
        if token_record is None:
            return
        token_record.revoked_at = utcnow()
        session.flush()

    def get_current_user(self, session: Session, access_token: str) -> User:
        payload = self._decode_token(access_token, expected_type="access")
        user_id = int(payload.get("sub", 0))
        user = session.get(User, user_id)
        if user is None or user.status != 1:
            raise AuthenticationError("当前登录用户不存在或已被禁用。")
        return user

    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        return pwd_context.verify(password, password_hash)

    def _build_access_token(self, user: User) -> str:
        expires_at = utcnow() + timedelta(
            minutes=self.settings.jwt_access_token_expire_minutes
        )
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "type": "access",
            "exp": expires_at,
        }
        return jwt.encode(
            payload,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
        )

    def _build_refresh_token(self, session: Session, user: User) -> str:
        expires_at = utcnow() + timedelta(
            days=self.settings.jwt_refresh_token_expire_days
        )
        token_jti = str(uuid4())
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "type": "refresh",
            "jti": token_jti,
            "exp": expires_at,
        }
        token = jwt.encode(
            payload,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
        )
        token_record = RefreshToken(
            user_id=user.id,
            token_jti=token_jti,
            token_hash=self._hash_token(token),
            expires_at=expires_at,
            created_at=utcnow(),
        )
        session.add(token_record)
        session.flush()
        return token

    def _decode_token(self, token: str, *, expected_type: str) -> dict:
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[self.settings.jwt_algorithm],
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError("Token 无效或已过期。") from exc

        token_type = str(payload.get("type", ""))
        if token_type != expected_type:
            raise AuthenticationError("Token 类型不匹配。")
        return payload

    def _hash_token(self, token: str) -> str:
        return sha256(token.encode("utf-8")).hexdigest()
