from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any, Generic, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

__all__ = [
    "Capability",
    "ChimericCompletionResponse",
    "ChimericFileUploadResponse",
    "ChimericStreamChunk",
    "CompletionResponse",
    "FileUploadResponse",
    "Input",
    "JSONSchemaArray",
    "JSONSchemaBoolean",
    "JSONSchemaInteger",
    "JSONSchemaNumber",
    "JSONSchemaObject",
    "JSONSchemaString",
    "JSONSchemaType",
    "Message",
    "Metadata",
    "ModelCapabilities",
    "ModelSummary",
    "NativeCompletionType",
    "NativeFileUploadType",
    "NativeStreamType",
    "Provider",
    "StreamChunk",
    "Tool",
    "ToolCall",
    "ToolCallChunk",
    "ToolExecutionResult",
    "ToolParameters",
    "Tools",
    "Usage",
]

# Generic type variables for native responses
NativeCompletionType = TypeVar("NativeCompletionType")
NativeStreamType = TypeVar("NativeStreamType")
NativeFileUploadType = TypeVar("NativeFileUploadType")

# Generic type aliases
Input: TypeAlias = str | list[Any]
Tools: TypeAlias = Iterable[Any] | None
Metadata: TypeAlias = dict[str, Any]


###################
# MESSAGE TYPES
###################


class Message(BaseModel):
    """Standardized message format for cross-provider compatibility.

    Attributes:
        role: The role of the message sender ("system", "user", "assistant", "tool").
        content: The message content, either as a string or list of content parts.
        name: Optional name identifier for the message sender.
        tool_calls: Optional list of tool calls made by the assistant.
        tool_call_id: Optional ID linking this message to a specific tool call.
    """

    role: str
    content: str | list[Any]
    name: str | None = None
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None


###################
# PROVIDER TYPES
###################


class Provider(Enum):
    """Supported LLM providers.

    Enum values correspond to the string identifiers used in configuration.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    GROK = "grok"
    GROQ = "groq"


###################
# MODEL TYPES
###################


class ModelCapabilities(TypedDict, total=False):
    """Capabilities advertised by a model.

    Attributes:
        streaming: Boolean indicating if the model supports streaming responses.
        function_calling: Boolean indicating if the model supports structured function/tool calling.
        vision: Boolean indicating if the model can process image inputs.
        tools: Boolean indicating if the model supports the modern external tool invocation system.
    """

    streaming: bool
    function_calling: bool
    vision: bool
    tools: bool


class Capability(BaseModel):
    """Features supported by an LLM provider or a specific model.

    Attributes:
        multimodal: Indicates if multiple input/output modalities (e.g., text, images) are supported.
        streaming: Indicates if responses can be streamed token-by-token.
        tools: Indicates if function/tool calling capabilities are supported.
        agents: Indicates if the provider supports autonomous agent functionalities.
        files: Indicates if file uploading, processing, and referencing are supported.
    """

    model_config = ConfigDict(frozen=True)

    multimodal: bool = False
    streaming: bool = False
    tools: bool = False
    agents: bool = False
    files: bool = False


class ModelSummary(BaseModel):
    """Concise summary of a model's attributes, often from a model listing operation.

    Attributes:
        id: Unique identifier for the model (e.g., "gpt-4", "claude-2").
        name: Human-readable name of the model.
        description: Optional brief description of the model's capabilities or purpose.
        owned_by: Optional identifier of the entity that owns or provides the model (e.g., "openai", "anthropic").
        created_at: Optional epoch timestamp (seconds since epoch) indicating when the model entry was created or made available.
        metadata: Optional dictionary for any other relevant model metadata not covered by specific fields.
        provider: Optional name or identifier of the LLM provider offering this model.
    """

    id: str
    name: str
    description: str | None = None
    owned_by: str | None = None
    created_at: int | None = None
    metadata: Metadata | None = None
    provider: str | None = None

    def __str__(self) -> str:
        """Return a string representation of the model summary."""
        return f"{self.name} ({self.provider})"


###################
# RESPONSE TYPES
###################


class Usage(BaseModel):
    """Token-usage summary for a completion request.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated in the response by the model.
        total_tokens: Total tokens processed, typically the sum of `prompt_tokens` and `completion_tokens`.
    """

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    """common response from a chat completion call.

    Attributes:
        content: The primary response content, typically text generated by the LLM.
        usage: Optional `Usage` object detailing token usage statistics for the completion.
        model: Optional string identifier of the model that generated the response.
        metadata: Optional `Metadata` containing additional provider-specific information about the response.
    """

    content: str | list[Any] = Field(default_factory=list)
    usage: Usage | None = None
    model: str | None = None
    metadata: Metadata | None = None

    def __str__(self) -> str:
        """Return a string representation of the completion response."""
        return (
            self.content
            if isinstance(self.content, str)
            else " ".join(str(c) for c in self.content if c)
        )


class ToolCall(BaseModel):
    """Represents a tool call with its essential information.

    Attributes:
        call_id: Provider-specific call identifier.
        name: Name of the tool being called.
        arguments: Arguments JSON string for the tool call.
    """

    call_id: str
    name: str
    arguments: str


class ToolCallChunk(BaseModel):
    """Information about a tool call in a streaming response.

    Attributes:
        id: Unique identifier for the tool call.
        call_id: Provider-specific call identifier.
        name: Name of the tool being called.
        arguments: Accumulated arguments JSON string.
        arguments_delta: Incremental change to arguments in this chunk.
        status: Current status of the tool call (e.g., 'started', 'arguments_streaming', 'completed').
    """

    id: str
    call_id: str | None = None
    name: str
    arguments: str = ""
    arguments_delta: str | None = None
    status: str | None = None


class StreamChunk(BaseModel):
    """Single chunk in a streaming chat response.

    Attributes:
        content: Text content of the current chunk. For some providers or configurations,
                 this might represent the full accumulated message up to this chunk.
        delta: Optional incremental text change in this chunk. If present, this is typically
               the new piece of text to append to the stream.
        finish_reason: Optional string indicating why the model stopped generating tokens,
                       usually sent with the final chunk of the stream.
        metadata: Optional `Metadata,` which may be sent with any chunk, or exclusively
                  with the final chunk, containing details like token counts or request IDs.
    """

    content: str | list[Any] = Field(default_factory=list)
    delta: str | None = None
    finish_reason: str | None = None
    metadata: Metadata | None = None

    def __str__(self) -> str:
        """Return a string representation of the stream chunk."""
        return self.delta if self.delta is not None else ""


class FileUploadResponse(BaseModel):
    """Information returned after uploading a file to a provider.

    Attributes:
        file_id: Unique identifier assigned to the uploaded file by the provider.
        filename: Original name of the file that was uploaded.
        bytes: Size of the uploaded file in bytes.
        purpose: The intended use of the file, as specified during upload (e.g., "assistants", "vision").
        created_at: Optional epoch timestamp (seconds since epoch) indicating when the file was uploaded/created.
        status: Optional current status of the file (e.g., "uploaded", "processing", "ready", "error").
        metadata: Optional dictionary for any additional provider-specific metadata about the file.
    """

    file_id: str
    filename: str
    bytes: int
    purpose: str | None = None
    created_at: int | None = None
    status: str | None = None
    metadata: dict[str, Any] | None = None


###################
# TOOL TYPES
###################


class ToolParameters(BaseModel):
    """JSON Schema for tool parameters with flexible additional properties."""

    model_config = ConfigDict(extra="allow")

    type: str = "object"
    strict: bool = True
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] | None = None
    additionalProperties: bool = False

    def model_dump(self, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        """Return dictionary representation, optionally excluding None values."""
        return super().model_dump(exclude_none=exclude_none, **kwargs)


class Tool(BaseModel):
    """Definition for a callable tool/function exposed to models.

    Attributes:
        name: Unique identifier for this tool, used by the model to specify which tool to call.
        description: Human-readable explanation of what the tool does, its purpose, and when to use it.
        parameters: Optional `ToolParameters` (JSON Schema) defining the arguments the tool accepts.
        function: Optional Python callable that implements the tool's functionality. Not serialized.
    """

    name: str
    description: str
    parameters: ToolParameters | None = None
    function: Callable[..., Any] | None = Field(default=None, exclude=True)

    def __str__(self) -> str:
        """Return a string representation of the tool."""
        return f"Tool({self.name}: {self.description[:50]}...)"


class ToolExecutionResult(BaseModel):
    """Result of executing a tool.

    Attributes:
        call_id: The ID of the tool call that was executed.
        name: The name of the tool that was executed.
        arguments: The arguments that were passed to the tool.
        result: The result of the tool execution, if successful.
        error: The error message, if execution failed.
        is_error: Boolean indicating if execution resulted in an error.
    """

    call_id: str
    name: str
    arguments: str

    result: str | None = None
    error: str | None = None
    is_error: bool = False


###################
# JSON SCHEMA TYPES
###################


class JSONSchemaType(TypedDict, total=False):
    """Base for JSON schema fragments, defining common properties.

    Attributes:
        type: The fundamental JSON schema data type (e.g., "string", "number", "object", "array", "boolean", "null").
        description: A human-readable explanation of the schema element's purpose.
        required: Indicates if this element is mandatory within its parent object schema.
    """

    type: str
    description: str
    required: bool


class JSONSchemaString(JSONSchemaType, total=False):
    """Schema for string values, extending JSONSchemaType.

    Attributes:
        enum: An optional list of allowed string values.
        format: An optional hint about the string's format (e.g., "date-time", "email", "uri", "uuid").
        default: An optional default value for the string if not provided.
        pattern: An optional regular expression that the string must match.
        minLength: An optional non-negative integer specifying the minimum string length.
        maxLength: An optional non-negative integer specifying the maximum string length.
        nullable: Boolean indicating if `null` is an allowed value for this string.
    """

    enum: list[str]
    format: str
    default: str
    pattern: str
    minLength: int
    maxLength: int
    nullable: bool


class JSONSchemaNumber(JSONSchemaType, total=False):
    """Schema for numeric values (float or double), extending JSONSchemaType.

    Attributes:
        minimum: An optional inclusive lower bound for the number.
        maximum: An optional inclusive upper bound for the number.
        exclusiveMinimum: An optional exclusive lower bound for the number.
        exclusiveMaximum: An optional exclusive upper bound for the number.
        multipleOf: An optional value that the number must be a multiple of.
        default: An optional default value for the number if not provided.
        nullable: Boolean indicating if `null` is an allowed value for this number.
    """

    minimum: float
    maximum: float
    exclusiveMinimum: float
    exclusiveMaximum: float
    multipleOf: float
    default: float
    nullable: bool


class JSONSchemaInteger(JSONSchemaType, total=False):
    """Schema for integer values, extending JSONSchemaType.

    Attributes:
        minimum: An optional inclusive lower bound for the integer.
        maximum: An optional inclusive upper bound for the integer.
        exclusiveMinimum: An optional exclusive lower bound for the integer.
        exclusiveMaximum: An optional exclusive upper bound for the integer.
        multipleOf: An optional value that the integer must be a multiple of.
        default: An optional default value for the integer if not provided.
        nullable: Boolean indicating if `null` is an allowed value for this integer.
    """

    minimum: int
    maximum: int
    exclusiveMinimum: int
    exclusiveMaximum: int
    multipleOf: int
    default: int
    nullable: bool


class JSONSchemaBoolean(JSONSchemaType, total=False):
    """Schema for boolean values, extending JSONSchemaType.

    Attributes:
        default: An optional default boolean value if not provided.
        nullable: Boolean indicating if `null` is an allowed value for this boolean field.
    """

    default: bool
    nullable: bool


class JSONSchemaArray(JSONSchemaType, total=False):
    """Schema for array values, extending JSONSchemaType.

    Attributes:
        items: A JSON Schema (or list of schemas for tuple validation) that defines the type of items in the array.
        minItems: An optional non-negative integer specifying the minimum number of items in the array.
        maxItems: An optional non-negative integer specifying the maximum number of items in the array.
        uniqueItems: An optional boolean indicating whether all items in the array must be unique.
        default: An optional default array value if not provided.
        nullable: Boolean indicating if `null` is an allowed value for this array.
    """

    items: dict[str, Any] | list[dict[str, Any]]
    minItems: int
    maxItems: int
    uniqueItems: bool
    default: list[Any]
    nullable: bool


class JSONSchemaObject(JSONSchemaType, total=False):
    """Schema for object values, extending JSONSchemaType.

    Attributes:
        properties: A dictionary mapping property names to their respective JSON Schema definitions.
        additionalProperties: A boolean or a JSON Schema that defines how additional properties (not in `properties`) are handled.
        default: An optional default object value if not provided.
        nullable: Boolean indicating if `null` is an allowed value for this object.
    """

    properties: dict[str, dict[str, Any]]
    additionalProperties: bool
    default: dict[str, Any]
    nullable: bool


###################
# COMBINED RESPONSE TYPES
###################


class ChimericCompletionResponse(BaseModel, Generic[NativeCompletionType]):
    """Combined response containing both native and common completion responses.

    This type allows methods to return both the provider's native response format and
    the common Chimeric format, giving users flexibility to use either as needed.

    Attributes:
        native: The provider's native completion response object.
        common: The common CompletionResponse in Chimeric's format.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    native: NativeCompletionType
    common: CompletionResponse


class ChimericStreamChunk(BaseModel, Generic[NativeStreamType]):
    """Combined stream chunk containing both native and common formats.

    This type provides access to both provider-specific stream chunk details and
    Chimeric's common format for streaming responses.

    Attributes:
        native: The provider's native stream chunk object.
        common: The common StreamChunk in Chimeric's format.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    native: NativeStreamType
    common: StreamChunk


class ChimericFileUploadResponse(BaseModel, Generic[NativeFileUploadType]):
    """Combined file upload response with both native and common formats.

    This type provides access to both provider-specific file upload details
    and Chimeric's common format.

    Attributes:
        native: The provider's native file upload response object.
        common: The common FileUploadResponse in Chimeric's format.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    native: NativeFileUploadType
    common: FileUploadResponse
