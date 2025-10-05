# app/api/v1/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

# This file defines the generic data structures for the API's requests and responses.

# --- Chat Schemas ---

class ChatMessageInput(BaseModel):
    """Represents a single message in a conversation history."""
    role: str = Field(
        ..., 
        description="Role of the sender, e.g., 'user' or 'assistant'.",
        examples=["user"]
    )
    content: str = Field(
        ...,
        description="Text content of the message.",
        examples=["Hello, what can you do?"]
    )
    
    # Provides a rich example for API documentation.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "Hi, who are you?"
            }
        }
    )

class ChatRequest(BaseModel):
    """Defines the request body for the main /chat endpoint."""
    query: str = Field(
        ..., 
        description="The user's current query to the assistant.",
        min_length=1,
        examples=["What is the capital of France?"]
    )
    history: List[ChatMessageInput] = Field(
        default_factory=list, 
        description="A list of previous messages in the conversation, oldest first."
    )

    # Provides a rich example for API documentation.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "And its population?",
                "history": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."}
                ]
            }
        }
    )

class ChatResponse(BaseModel):
    """Defines the response body from the /chat endpoint."""
    answer: str = Field(
        ..., 
        description="The chatbot's textual response."
    )
    error_message: Optional[str] = Field(
        None, 
        description="An error message if the request processing failed."
    )

    # Provides a rich example for API documentation.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "The population of Paris was approximately 2.1 million people as of 2023.",
                "error_message": None
            }
        }
    )

# --- Transcription Schemas ---

class TranscribeResponse(BaseModel):
    """Response model for the /transcribe endpoint."""
    transcript: Optional[str] = Field(None, description="The transcribed text from the audio.")
    error_message: Optional[str] = Field(None, description="An error message if transcription failed.")

    # Provides a rich example for API documentation.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transcript": "This is a test of the transcription service.",
                "error_message": None
            }
        }
    )

# --- Text-to-Speech Schemas ---

class SpeechRequest(BaseModel):
    """Request model for the /speech endpoint."""
    text: str = Field(
        ..., 
        min_length=1, 
        description="The text to be synthesized into speech.", 
        examples=["Hello world."]
    )
    instructions: Optional[str] = Field(
        None,
        description="Optional instructions for speech generation (e.g., tone, emotion), for compatible models.",
        examples=["Speak in a calm voice."]
    )

    # Provides a rich example for API documentation.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The message has been sent successfully.",
                "instructions": None
            }
        }
    )
