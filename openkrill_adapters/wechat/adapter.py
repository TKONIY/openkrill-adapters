"""WeChat Work (企业微信) Adapter — forward OpenKrill messages to/from WeCom.

This is an OUTBOUND adapter: it bridges an OpenKrill channel to a WeChat Work
(企业微信) application. Messages sent in the OpenKrill channel are forwarded
to WeCom users/departments, and inbound callback messages from WeCom are
forwarded back to OpenKrill.

企业微信 API 文档: https://developer.work.weixin.qq.com/document/path/90664

Config (adapter_config):
    corp_id: 企业 ID (企业微信管理后台 → 我的企业 → 企业信息)
    corp_secret: 应用的 Secret (应用管理 → 应用 → Secret)
    agent_id: 企业应用 AgentId (应用管理 → 应用 → AgentId)
    token: 回调 Token (应用管理 → 接收消息 → Token)
    encoding_aes_key: 回调 EncodingAESKey (应用管理 → 接收消息 → EncodingAESKey)
    to_user: 发送目标用户 (默认 "@all"，即所有人)
    to_party: 发送目标部门 ID (可选)
    to_tag: 发送目标标签 ID (可选)

No external library needed — uses httpx for HTTP requests and stdlib for
message encryption/decryption.

Requires: pip install httpx
    (httpx is already a dependency of openkrill-adapters)
"""

import base64
import hashlib
import logging
import socket
import struct
import time
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import httpx

from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    StreamChunk,
)

logger = logging.getLogger(__name__)

# ── 企业微信 API Base URL ──
QYAPI_BASE = "https://qyapi.weixin.qq.com/cgi-bin"


class WXBizMsgCrypt:
    """企业微信消息加解密工具 (simplified).

    Implements the WXBizMsgCrypt protocol for verifying and decrypting
    callback messages from WeChat Work.

    Full specification:
    https://developer.work.weixin.qq.com/document/path/90930

    加解密流程:
    1. 验证签名 (msg_signature) — SHA1(sorted([token, timestamp, nonce, encrypted_msg]))
    2. AES-256-CBC 解密 (key = Base64Decode(encoding_aes_key + "="))
    3. 去除 PKCS#7 填充
    4. 解析: random(16B) + msg_len(4B, network byte order) + msg + corp_id

    NOTE: This is a simplified implementation suitable for basic text message
    handling. For production use with media messages, consider using the
    official WXBizMsgCrypt library from WeChat Work.
    企业微信官方提供了完整的加解密库，生产环境建议使用官方库。
    """

    def __init__(self, token: str, encoding_aes_key: str, corp_id: str) -> None:
        self.token = token
        self.corp_id = corp_id
        # encoding_aes_key is Base64 encoded (43 chars), pad with "=" to decode
        # encoding_aes_key 是 43 字符的 Base64 编码，补 "=" 后解码为 32 字节 AES key
        self.aes_key = base64.b64decode(encoding_aes_key + "=")
        if len(self.aes_key) != 32:
            raise ValueError(
                f"encoding_aes_key must decode to 32 bytes, got {len(self.aes_key)}"
            )

    def verify_signature(
        self, msg_signature: str, timestamp: str, nonce: str, encrypted_msg: str
    ) -> bool:
        """Verify the callback message signature.
        验证回调消息签名，防止伪造请求。

        Args:
            msg_signature: 签名串 (URL 参数 msg_signature)
            timestamp: 时间戳 (URL 参数 timestamp)
            nonce: 随机串 (URL 参数 nonce)
            encrypted_msg: 加密的消息体 (XML 中的 Encrypt 字段)
        """
        sort_list = sorted([self.token, timestamp, nonce, encrypted_msg])
        raw = "".join(sort_list)
        expected = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return expected == msg_signature

    def decrypt_message(self, encrypted_msg: str) -> str:
        """Decrypt an encrypted callback message.
        解密企业微信回调消息。

        AES-256-CBC decryption with PKCS#7 padding removal.
        Validates that the trailing corp_id matches our configured corp_id.

        Args:
            encrypted_msg: Base64-encoded encrypted message (XML Encrypt field)

        Returns:
            Decrypted XML message content (plain text)
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as e:
            raise ImportError(
                "cryptography is required for WeChat Work message decryption. "
                "Install with: pip install cryptography"
            ) from e

        # Base64 decode the encrypted message
        # Base64 解码加密消息
        encrypted_bytes = base64.b64decode(encrypted_msg)

        # AES-256-CBC decrypt, IV is the first 16 bytes of the key
        # AES-256-CBC 解密，IV 取 key 的前 16 字节
        iv = self.aes_key[:16]
        cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(encrypted_bytes) + decryptor.finalize()

        # Remove PKCS#7 padding
        # 去除 PKCS#7 填充
        pad_len = decrypted[-1]
        decrypted = decrypted[:-pad_len]

        # Parse: random(16B) + msg_len(4B big-endian) + msg + corp_id
        # 解析: 随机字符串(16字节) + 消息长度(4字节网络字节序) + 消息内容 + 企业ID
        msg_len = struct.unpack("!I", decrypted[16:20])[0]
        msg_content = decrypted[20 : 20 + msg_len].decode("utf-8")
        trailing_corp_id = decrypted[20 + msg_len :].decode("utf-8")

        if trailing_corp_id != self.corp_id:
            raise ValueError(
                f"corp_id mismatch: expected {self.corp_id}, got {trailing_corp_id}"
            )

        return msg_content

    def encrypt_message(self, reply_msg: str, timestamp: str, nonce: str) -> str:
        """Encrypt a reply message for WeChat Work callback response.
        加密回复消息（用于被动回复）。

        Args:
            reply_msg: Plain text reply XML
            timestamp: Timestamp string
            nonce: Nonce string

        Returns:
            Encrypted XML response string
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as e:
            raise ImportError(
                "cryptography is required for WeChat Work message encryption. "
                "Install with: pip install cryptography"
            ) from e

        import os

        msg_bytes = reply_msg.encode("utf-8")
        corp_id_bytes = self.corp_id.encode("utf-8")

        # Build plaintext: random(16B) + msg_len(4B) + msg + corp_id
        # 构建明文: 随机串(16字节) + 消息长度(4字节) + 消息 + 企业ID
        random_bytes = os.urandom(16)
        msg_len_bytes = struct.pack("!I", len(msg_bytes))
        plaintext = random_bytes + msg_len_bytes + msg_bytes + corp_id_bytes

        # PKCS#7 padding to AES block size (16 bytes)
        # PKCS#7 填充到 AES 块大小 (16 字节)
        block_size = 16
        pad_len = block_size - (len(plaintext) % block_size)
        plaintext += bytes([pad_len]) * pad_len

        # AES-256-CBC encrypt
        iv = self.aes_key[:16]
        cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(plaintext) + encryptor.finalize()

        encrypted_msg = base64.b64encode(encrypted).decode("utf-8")

        # Generate signature
        # 生成签名
        sort_list = sorted([self.token, timestamp, nonce, encrypted_msg])
        signature = hashlib.sha1("".join(sort_list).encode("utf-8")).hexdigest()

        # Build response XML
        # 构建响应 XML
        return (
            f"<xml>"
            f"<Encrypt><![CDATA[{encrypted_msg}]]></Encrypt>"
            f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
            f"<TimeStamp>{timestamp}</TimeStamp>"
            f"<Nonce><![CDATA[{nonce}]]></Nonce>"
            f"</xml>"
        )


class WeChatWorkAdapter(BaseAdapter):
    """Outbound adapter that bridges OpenKrill channels to WeChat Work (企业微信).

    Uses 企业微信 server API to send messages, and provides webhook callback
    handling for receiving inbound messages.

    企业微信消息收发流程:
    - 发送: 调用 /message/send 接口主动推送消息给用户
    - 接收: 企业微信通过 HTTP 回调推送用户消息到配置的回调 URL

    Lifecycle:
        1. __init__(config) — parse corp_id, corp_secret, agent_id, etc.
        2. connect() — fetch access_token from 企业微信
        3. send() — forward message via /message/send API
        4. send_stream() — buffer and send (企业微信 has no streaming API)
        5. start_listening() — set up webhook callback server
        6. disconnect() — clean up HTTP client and callback server
    """

    # 企业微信 text message length limit (2048 bytes)
    # 企业微信文本消息最大长度 2048 字节
    MAX_MESSAGE_LENGTH = 2048

    # access_token validity period (7200 seconds = 2 hours)
    # access_token 有效期 7200 秒（2 小时）
    TOKEN_EXPIRY_SECONDS = 7200

    # Refresh token 5 minutes before expiry to avoid edge cases
    # 提前 5 分钟刷新 token，避免边界情况
    TOKEN_REFRESH_MARGIN = 300

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Required config fields / 必需配置项
        self._corp_id: str = config.get("corp_id", "")
        self._corp_secret: str = config.get("corp_secret", "")
        self._agent_id: int = int(config.get("agent_id", 0))
        self._token: str = config.get("token", "")
        self._encoding_aes_key: str = config.get("encoding_aes_key", "")

        # Optional: message target / 可选：消息发送目标
        self._to_user: str = config.get("to_user", "@all")
        self._to_party: str = config.get("to_party", "")
        self._to_tag: str = config.get("to_tag", "")

        # Runtime state / 运行时状态
        self._access_token: str = ""
        self._token_expires_at: float = 0.0  # Unix timestamp
        self._http_client: httpx.AsyncClient | None = None
        self._crypto: WXBizMsgCrypt | None = None
        self._on_message_callback: Any = None
        self._callback_server: Any = None  # aiohttp server for webhook callbacks

        # Validate required fields / 验证必填项
        if not self._corp_id:
            raise ValueError("WeChat Work adapter requires 'corp_id' in config")
        if not self._corp_secret:
            raise ValueError("WeChat Work adapter requires 'corp_secret' in config")
        if not self._agent_id:
            raise ValueError("WeChat Work adapter requires 'agent_id' in config")

    async def connect(self) -> None:
        """Initialize HTTP client and fetch access_token.
        初始化 HTTP 客户端并获取 access_token。

        企业微信 access_token 获取:
        GET https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=ID&corpsecret=SECRET
        返回: {"errcode": 0, "errmsg": "ok", "access_token": "...", "expires_in": 7200}
        """
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Initialize message crypto if callback config is provided
        # 如果提供了回调配置，初始化消息加解密工具
        if self._token and self._encoding_aes_key:
            self._crypto = WXBizMsgCrypt(
                token=self._token,
                encoding_aes_key=self._encoding_aes_key,
                corp_id=self._corp_id,
            )

        # Fetch initial access_token
        # 获取初始 access_token
        await self._refresh_access_token()

        logger.info(
            "WeChat Work adapter connected (corp_id=%s, agent_id=%s)",
            self._corp_id,
            self._agent_id,
        )

    async def disconnect(self) -> None:
        """Clean up HTTP client and stop callback server.
        清理 HTTP 客户端和回调服务器。
        """
        if self._callback_server is not None:
            try:
                await self._callback_server.cleanup()
            except Exception:
                logger.debug("Error closing callback server", exc_info=True)
            self._callback_server = None

        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

        self._access_token = ""
        self._token_expires_at = 0.0
        self._crypto = None
        logger.info("WeChat Work adapter disconnected")

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Forward the last message to WeChat Work users.
        将最新消息推送给企业微信用户。

        Uses the /message/send API endpoint:
        POST https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=TOKEN

        Request body (text message):
        {
            "touser": "@all",
            "msgtype": "text",
            "agentid": 1000002,
            "text": {"content": "消息内容"}
        }

        Response: {"errcode": 0, "errmsg": "ok", ...}
        """
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized. Call connect() first.")

        message = self._pick_outbound_message(messages)
        if not message:
            return AdapterResponse(content="(no message to forward)", content_type="text")

        text = message.content
        sent = await self._send_text_message(text)
        return AdapterResponse(
            content=sent,
            content_type="text",
            metadata={"wechat_agent_id": self._agent_id},
        )

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        """Buffer all chunks and send as a single message.
        缓冲所有内容后作为单条消息发送（企业微信不支持流式推送）。

        WeChat Work has no streaming message API, so we send a complete message.
        """
        response = await self.send(messages)
        yield StreamChunk(type="text", content=response.content)

    async def health_check(self) -> bool:
        """Verify the access_token is valid by calling a lightweight API.
        通过调用轻量级 API 验证 access_token 是否有效。

        Uses /ip/list endpoint (returns WeChat Work server IP list).
        """
        if self._http_client is None:
            return False
        try:
            token = await self._get_access_token()
            resp = await self._http_client.get(
                f"{QYAPI_BASE}/get_api_domain_ip",
                params={"access_token": token},
            )
            data = resp.json()
            ok = data.get("errcode", -1) == 0
            logger.debug("WeChat Work health check: errcode=%s", data.get("errcode"))
            return ok
        except Exception:
            logger.exception("WeChat Work health check failed")
            return False

    @property
    def adapter_type(self) -> str:
        return "wechat-work"

    @property
    def max_capability(self) -> AdapterCapability:
        # 企业微信支持文本、图片、文件等，但目前只实现文本
        # WeCom supports text, images, files, etc. — currently text only
        return AdapterCapability.L0_TEXT

    # ── Inbound: receive messages from WeChat Work callback / 接收企业微信回调消息 ──

    def set_on_message(self, callback: Any) -> None:
        """Register a callback for inbound WeChat Work messages.
        注册接收企业微信消息的回调函数。

        Callback signature:
            async def on_message(text: str, sender_name: str, metadata: dict) -> None
        """
        self._on_message_callback = callback

    async def start_listening(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start an HTTP callback server to receive WeChat Work messages.
        启动 HTTP 回调服务器，接收企业微信推送的消息。

        企业微信回调配置:
        1. 在企业微信管理后台 → 应用管理 → 应用 → 接收消息 → 设置 API 接收
        2. URL: http://your-server:8080/wechat/callback
        3. Token 和 EncodingAESKey 填入 adapter_config

        回调验证 (GET 请求):
            企业微信发送 GET 请求验证 URL 有效性，参数包含 msg_signature, timestamp,
            nonce, echostr。服务器需解密 echostr 并返回明文。

        消息接收 (POST 请求):
            企业微信 POST 加密的 XML 消息体，服务器需验证签名、解密、解析。

        Args:
            host: Callback server bind address (default 0.0.0.0)
            port: Callback server port (default 8080)

        NOTE: In production, you would typically put this behind a reverse proxy
        (nginx) with HTTPS. The callback URL configured in 企业微信 must be HTTPS.
        生产环境建议使用 nginx 反向代理并配置 HTTPS。
        """
        if self._crypto is None:
            raise RuntimeError(
                "Message crypto not initialized. "
                "Provide 'token' and 'encoding_aes_key' in config to enable callbacks."
            )

        try:
            from aiohttp import web
        except ImportError as e:
            raise ImportError(
                "aiohttp is required for WeChat Work callback server. "
                "Install with: pip install aiohttp"
            ) from e

        app = web.Application()
        app.router.add_get("/wechat/callback", self._handle_verify)
        app.router.add_post("/wechat/callback", self._handle_callback)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        self._callback_server = runner

        logger.info(
            "WeChat Work callback server started at http://%s:%d/wechat/callback",
            host,
            port,
        )

    async def _handle_verify(self, request: Any) -> Any:
        """Handle GET request for URL verification (回调 URL 验证).

        企业微信在配置回调 URL 时会发送 GET 请求验证:
        GET /wechat/callback?msg_signature=xxx&timestamp=xxx&nonce=xxx&echostr=xxx

        服务器需要:
        1. 验证 msg_signature
        2. 解密 echostr
        3. 返回解密后的明文 echostr
        """
        from aiohttp import web

        msg_signature = request.query.get("msg_signature", "")
        timestamp = request.query.get("timestamp", "")
        nonce = request.query.get("nonce", "")
        echostr = request.query.get("echostr", "")

        if not self._crypto:
            return web.Response(status=403, text="Crypto not configured")

        # Verify signature / 验证签名
        if not self._crypto.verify_signature(msg_signature, timestamp, nonce, echostr):
            logger.warning("WeChat Work callback verification failed: bad signature")
            return web.Response(status=403, text="Invalid signature")

        # Decrypt echostr and return plaintext / 解密 echostr 并返回明文
        try:
            decrypted = self._crypto.decrypt_message(echostr)
            logger.info("WeChat Work callback URL verified successfully")
            return web.Response(text=decrypted)
        except Exception:
            logger.exception("Failed to decrypt echostr")
            return web.Response(status=500, text="Decryption failed")

    async def _handle_callback(self, request: Any) -> Any:
        """Handle POST request for inbound messages (接收消息回调).

        企业微信 POST 的消息格式 (XML):
        <xml>
            <ToUserName><![CDATA[企业号CorpID]]></ToUserName>
            <Encrypt><![CDATA[加密消息体]]></Encrypt>
            <AgentID>应用ID</AgentID>
        </xml>

        解密后的消息 XML (文本消息):
        <xml>
            <ToUserName><![CDATA[CorpID]]></ToUserName>
            <FromUserName><![CDATA[UserID]]></FromUserName>
            <CreateTime>1348831860</CreateTime>
            <MsgType><![CDATA[text]]></MsgType>
            <Content><![CDATA[消息内容]]></Content>
            <MsgId>1234567890123456</MsgId>
            <AgentID>1000002</AgentID>
        </xml>
        """
        from aiohttp import web

        msg_signature = request.query.get("msg_signature", "")
        timestamp = request.query.get("timestamp", "")
        nonce = request.query.get("nonce", "")

        if not self._crypto:
            return web.Response(status=403, text="Crypto not configured")

        try:
            # Parse the outer XML to get the encrypted message
            # 解析外层 XML 获取加密消息体
            body = await request.text()
            root = ET.fromstring(body)
            encrypted_msg = root.findtext("Encrypt", "")

            if not encrypted_msg:
                logger.warning("WeChat Work callback: no Encrypt field in XML")
                return web.Response(status=400, text="Missing Encrypt field")

            # Verify signature / 验证签名
            if not self._crypto.verify_signature(
                msg_signature, timestamp, nonce, encrypted_msg
            ):
                logger.warning("WeChat Work callback: bad signature")
                return web.Response(status=403, text="Invalid signature")

            # Decrypt the message / 解密消息
            decrypted_xml = self._crypto.decrypt_message(encrypted_msg)
            msg_root = ET.fromstring(decrypted_xml)

            # Extract message fields / 提取消息字段
            msg_type = msg_root.findtext("MsgType", "")
            from_user = msg_root.findtext("FromUserName", "")
            content = msg_root.findtext("Content", "")
            msg_id = msg_root.findtext("MsgId", "")
            agent_id = msg_root.findtext("AgentID", "")
            create_time = msg_root.findtext("CreateTime", "")

            logger.info(
                "WeChat Work callback: type=%s, from=%s, agent=%s",
                msg_type,
                from_user,
                agent_id,
            )

            # Currently only handle text messages / 目前只处理文本消息
            if msg_type == "text" and content and self._on_message_callback:
                await self._on_message_callback(
                    text=content.strip(),
                    sender_name=from_user,  # 企业微信 UserID
                    metadata={
                        "wechat_user_id": from_user,
                        "wechat_msg_type": msg_type,
                        "wechat_msg_id": msg_id,
                        "wechat_agent_id": agent_id,
                        "wechat_create_time": create_time,
                    },
                )
            elif msg_type != "text":
                logger.debug(
                    "WeChat Work callback: ignoring non-text message type=%s", msg_type
                )

            # Return success (企业微信要求返回 "success" 或空字符串)
            return web.Response(text="success")

        except ET.ParseError:
            logger.exception("WeChat Work callback: XML parse error")
            return web.Response(status=400, text="Invalid XML")
        except Exception:
            logger.exception("WeChat Work callback: unexpected error")
            return web.Response(status=500, text="Internal error")

    # ── Token management / access_token 管理 ──

    async def _get_access_token(self) -> str:
        """Get a valid access_token, refreshing if expired.
        获取有效的 access_token，过期则自动刷新。

        企业微信 access_token 有效期 2 小时，需要缓存并在过期前刷新。
        """
        now = time.time()
        if self._access_token and now < (
            self._token_expires_at - self.TOKEN_REFRESH_MARGIN
        ):
            return self._access_token
        await self._refresh_access_token()
        return self._access_token

    async def _refresh_access_token(self) -> None:
        """Fetch a new access_token from 企业微信 API.
        从企业微信 API 获取新的 access_token。

        GET https://qyapi.weixin.qq.com/cgi-bin/gettoken
            ?corpid=CORP_ID&corpsecret=CORP_SECRET

        Response:
            {"errcode": 0, "errmsg": "ok", "access_token": "...", "expires_in": 7200}

        Common errors / 常见错误:
            40001: 不合法的 secret
            40013: 不合法的 CorpID
            40056: 不合法的 agentid
        """
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")

        resp = await self._http_client.get(
            f"{QYAPI_BASE}/gettoken",
            params={"corpid": self._corp_id, "corpsecret": self._corp_secret},
        )
        data = resp.json()

        errcode = data.get("errcode", -1)
        if errcode != 0:
            errmsg = data.get("errmsg", "unknown error")
            raise RuntimeError(
                f"Failed to get WeChat Work access_token: "
                f"errcode={errcode}, errmsg={errmsg}"
            )

        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", self.TOKEN_EXPIRY_SECONDS)
        self._token_expires_at = time.time() + expires_in
        logger.info(
            "WeChat Work access_token refreshed (expires_in=%ds)", expires_in
        )

    # ── Message sending / 消息发送 ──

    async def _send_text_message(self, text: str) -> str:
        """Send a text message via 企业微信 /message/send API.
        通过企业微信 API 发送文本消息。

        POST https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=TOKEN
        Body:
        {
            "touser": "UserID1|UserID2" or "@all",
            "toparty": "PartyID1|PartyID2",
            "totag": "TagID1|TagID2",
            "msgtype": "text",
            "agentid": 1000002,
            "text": {"content": "消息内容"},
            "safe": 0,           // 是否保密消息 (0=否, 1=是)
            "enable_duplicate_check": 0
        }

        Response:
            {"errcode": 0, "errmsg": "ok", "invaliduser": "", ...}

        Text messages are limited to 2048 bytes. We split longer messages.
        文本消息限制 2048 字节，超长消息会被自动分片发送。
        """
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")

        chunks = self._split_text(text, self.MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            token = await self._get_access_token()
            payload = {
                "msgtype": "text",
                "agentid": self._agent_id,
                "text": {"content": chunk},
                "safe": 0,
            }

            # Set message target / 设置消息发送目标
            if self._to_user:
                payload["touser"] = self._to_user
            if self._to_party:
                payload["toparty"] = self._to_party
            if self._to_tag:
                payload["totag"] = self._to_tag

            resp = await self._http_client.post(
                f"{QYAPI_BASE}/message/send",
                params={"access_token": token},
                json=payload,
            )
            data = resp.json()
            errcode = data.get("errcode", -1)

            if errcode == 40014 or errcode == 42001:
                # access_token expired or invalid — retry with fresh token
                # access_token 过期或无效，刷新后重试
                logger.warning(
                    "WeChat Work access_token invalid (errcode=%d), refreshing...",
                    errcode,
                )
                await self._refresh_access_token()
                token = self._access_token
                resp = await self._http_client.post(
                    f"{QYAPI_BASE}/message/send",
                    params={"access_token": token},
                    json=payload,
                )
                data = resp.json()
                errcode = data.get("errcode", -1)

            if errcode != 0:
                errmsg = data.get("errmsg", "unknown error")
                logger.error(
                    "WeChat Work send failed: errcode=%d, errmsg=%s", errcode, errmsg
                )
                raise RuntimeError(
                    f"WeChat Work message send failed: errcode={errcode}, errmsg={errmsg}"
                )

            invalid_user = data.get("invaliduser", "")
            if invalid_user:
                logger.warning("WeChat Work: invalid users: %s", invalid_user)

        return text

    # ── Internal helpers / 内部工具方法 ──

    @staticmethod
    def _pick_outbound_message(messages: list[AdapterMessage]) -> AdapterMessage | None:
        """Pick the message to forward outbound (latest assistant, then user).
        选择要转发的消息（优先最新的 assistant 消息，其次 user 消息）。
        """
        for m in reversed(messages):
            if m.role == "assistant":
                return m
        for m in reversed(messages):
            if m.role == "user":
                return m
        return None

    @staticmethod
    def _split_text(text: str, max_length: int) -> list[str]:
        """Split text into chunks that fit within 企业微信's message limit.
        将文本分片以满足企业微信消息长度限制。

        Uses byte length since 企业微信 counts bytes (UTF-8 encoded).
        企业微信按字节计算长度，中文字符占 3 字节。
        """
        # For simplicity, use character-based splitting with conservative limit
        # 简化实现：使用字符级分割，保守估计（中文3字节/字，留余量）
        safe_limit = max_length // 3  # Conservative for CJK text
        if len(text.encode("utf-8")) <= max_length:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text.encode("utf-8")) <= max_length:
                chunks.append(text)
                break
            # Try to split at a newline for readability
            # 优先在换行符处分割，保持可读性
            split_pos = text.rfind("\n", 0, safe_limit)
            if split_pos == -1:
                split_pos = safe_limit
            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip("\n")
        return chunks
