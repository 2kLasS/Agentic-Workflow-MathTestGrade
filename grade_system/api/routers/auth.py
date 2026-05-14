from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from grade_system.api import deps
from grade_system.api.schemas.auth import (
    AccessTokenResponse,
    CurrentUserResponse,
    LoginRequest,
    LogoutRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
)
from grade_system.application.auth_service import (
    AuthService,
    AuthenticationError,
    ConflictError,
)
from grade_system.persistence import get_db_session
from grade_system.persistence.models import User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    session: Session = Depends(get_db_session),
) -> TokenResponse:
    auth_service = AuthService()
    try:
        user = auth_service.register_user(
            session,
            username=payload.username,
            password=payload.password,
            display_name=payload.display_name,
        )
        tokens = auth_service.issue_tokens(session, user)
        session.commit()
        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type="bearer",
            expires_in=tokens.expires_in,
            user=CurrentUserResponse.model_validate(user, from_attributes=True),
        )
    except ConflictError as exc:
        session.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    session: Session = Depends(get_db_session),
) -> TokenResponse:
    auth_service = AuthService()
    try:
        user = auth_service.authenticate_user(
            session,
            username=payload.username,
            password=payload.password,
        )
        tokens = auth_service.issue_tokens(session, user)
        session.commit()
        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type="bearer",
            expires_in=tokens.expires_in,
            user=CurrentUserResponse.model_validate(user, from_attributes=True),
        )
    except AuthenticationError as exc:
        session.rollback()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.post("/refresh", response_model=AccessTokenResponse)
def refresh_access_token(
    payload: RefreshTokenRequest,
    session: Session = Depends(get_db_session),
) -> AccessTokenResponse:
    auth_service = AuthService()
    try:
        access_token = auth_service.refresh_access_token(session, payload.refresh_token)
        return AccessTokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=auth_service.settings.jwt_access_token_expire_minutes * 60,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    payload: LogoutRequest,
    session: Session = Depends(get_db_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    _ = current_user
    auth_service = AuthService()
    try:
        auth_service.revoke_refresh_token(session, payload.refresh_token)
        session.commit()
    except AuthenticationError as exc:
        session.rollback()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.get("/me", response_model=CurrentUserResponse)
def get_me(current_user: User = Depends(deps.get_current_user)) -> CurrentUserResponse:
    return CurrentUserResponse.model_validate(current_user, from_attributes=True)
