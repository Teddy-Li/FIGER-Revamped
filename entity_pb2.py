# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: entity.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x65ntity.proto\"\xf7\x01\n\x07Mention\x12\r\n\x05start\x18\x01 \x01(\x05\x12\x0b\n\x03\x65nd\x18\x02 \x01(\x05\x12\x0e\n\x06tokens\x18\x03 \x03(\t\x12\x10\n\x08pos_tags\x18\x04 \x03(\t\x12!\n\x04\x64\x65ps\x18\x05 \x03(\x0b\x32\x13.Mention.Dependency\x12\x13\n\x0b\x65ntity_name\x18\x06 \x01(\t\x12\x10\n\x08\x66\x65\x61tures\x18\x07 \x03(\t\x12\x0e\n\x06labels\x18\x08 \x03(\t\x12\x0e\n\x06sentid\x18\t \x01(\x05\x12\x0e\n\x06\x66ileid\x18\n \x01(\t\x1a\x34\n\nDependency\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x0b\n\x03gov\x18\x02 \x01(\x05\x12\x0b\n\x03\x64\x65p\x18\x03 \x01(\x05')


_MENTION = DESCRIPTOR.message_types_by_name['Mention']
_MENTION_DEPENDENCY = _MENTION.nested_types_by_name['Dependency']
Mention = _reflection.GeneratedProtocolMessageType('Mention', (_message.Message,), {

  'Dependency' : _reflection.GeneratedProtocolMessageType('Dependency', (_message.Message,), {
    'DESCRIPTOR' : _MENTION_DEPENDENCY,
    '__module__' : 'entity_pb2'
    # @@protoc_insertion_point(class_scope:Mention.Dependency)
    })
  ,
  'DESCRIPTOR' : _MENTION,
  '__module__' : 'entity_pb2'
  # @@protoc_insertion_point(class_scope:Mention)
  })
_sym_db.RegisterMessage(Mention)
_sym_db.RegisterMessage(Mention.Dependency)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MENTION._serialized_start=17
  _MENTION._serialized_end=264
  _MENTION_DEPENDENCY._serialized_start=212
  _MENTION_DEPENDENCY._serialized_end=264
# @@protoc_insertion_point(module_scope)