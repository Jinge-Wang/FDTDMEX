"""``SceneModel`` — a pydantic facade over fdtdx's ``JsonSetup`` (the small, editable config).

This is the **hybrid** schema: pydantic owns the clean, validated, JSON-round-tripping surface that
the reactive window (pydantic-ai + AG-UI + FastAPI) and the MCP tools edit, while the proven
per-object encoders in :mod:`fdtdx.conversion.json` own the actual (de)serialization of every physics
object. We do **not** reimplement a pydantic class per object type; instead each object/constraint is
carried as its fdtdx export tree (``{"__module__", "__name__", ...public fields...}``), which is
lossless and directly reconstructible.

The model exposes lightweight typed conveniences (object/constraint summaries, counts) so a UI can
render and edit the setup, and lowers to a ``JsonSetup`` (→ ``place_objects``) via
:meth:`to_json_setup`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fdtdx.conversion.json import JsonSetup


class ObjectSummary(BaseModel):
    """A compact, UI-friendly view of one object/constraint in the setup."""

    type: str
    name: str | None = None


class SceneModel(BaseModel):
    """Editable, validated, JSON-round-tripping mirror of a ``JsonSetup``.

    Fields hold fdtdx export trees (lossless); use :meth:`to_json_setup` to lower to live objects and
    :meth:`from_json_setup` / :meth:`from_scene` to lift. ``model_dump_json`` / ``model_validate_json``
    give the small editable JSON the reactive window saves.
    """

    schema_version: str = Field(default="fdtdx.place_objects.v1")
    fdtdx_version: str = Field(default="0.6.2")
    #: fdtdx export tree of the SimulationConfig.
    config: dict[str, Any]
    #: fdtdx export trees of the object_list entries (Volume, materials, sources, detectors, boundaries).
    object_list: list[dict[str, Any]] = Field(default_factory=list)
    #: fdtdx export trees of the placement constraints.
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] | None = None

    # ----- lifting / lowering -------------------------------------------------------------------
    @classmethod
    def from_json_setup(cls, setup: "JsonSetup") -> "SceneModel":
        """Build a :class:`SceneModel` from a ``JsonSetup`` (export each piece to its fdtdx tree)."""
        from fdtdx.conversion.json import export_json

        return cls(
            config=export_json(setup.config),
            object_list=[export_json(o) for o in setup.object_list],
            constraints=[export_json(c) for c in setup.constraints],
            meta=setup.meta,
        )

    @classmethod
    def from_scene(cls, scene: Any) -> "SceneModel":
        """Build a :class:`SceneModel` from a :class:`fdtdx.Scene`."""
        return cls.from_json_setup(scene.to_json_setup())

    def to_json_setup(self, validate: bool = True) -> "JsonSetup":
        """Lower back to a live ``JsonSetup`` (reconstruct config + objects + constraints)."""
        from fdtdx.conversion.json import JsonSetup, _import_obj_from_json

        config = _import_obj_from_json(self.config)
        objects = [_import_obj_from_json(o) for o in self.object_list]
        constraints = [_import_obj_from_json(c) for c in self.constraints]
        js = JsonSetup(config=config, object_list=objects, constraints=constraints, meta=self.meta)
        if validate:
            js.validate()
        return js

    def to_scene(self):
        """Lower to a :class:`fdtdx.Scene`."""
        from fdtdx.scene import Scene

        return Scene.from_json_setup(self.to_json_setup())

    # ----- UI conveniences ----------------------------------------------------------------------
    @staticmethod
    def _summ(tree: dict[str, Any]) -> ObjectSummary:
        name = tree.get("name")
        return ObjectSummary(type=str(tree.get("__name__", "?")), name=name if isinstance(name, str) else None)

    @property
    def objects(self) -> list[ObjectSummary]:
        """Compact ``(type, name)`` summaries of every object — for UI listing/selection."""
        return [self._summ(o) for o in self.object_list]

    @property
    def n_objects(self) -> int:
        return len(self.object_list)

    def describe(self) -> dict[str, Any]:
        """A small dict an agent/UI can read: counts + object/constraint type lists."""
        return {
            "n_objects": len(self.object_list),
            "n_constraints": len(self.constraints),
            "objects": [s.model_dump() for s in self.objects],
            "constraint_types": [str(c.get("__name__", "?")) for c in self.constraints],
        }
