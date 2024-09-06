import {
  usePlayer,
  usePlayers,
  useRound,
  useStage,
} from "@empirica/core/player/classic/react";
import { Loading } from "@empirica/core/player/react";
import React from "react";
import { SharedStage } from "./stages/SharedStage";
import { SharedStage_2 } from "./stages/SharedStage_2.jsx";
import { SharedStage_3 } from "./stages/SharedStage_3.jsx";
import { Instructions } from "./stages/Instructions.jsx";
import { EndOfTests } from "./stages/EndOfTests.jsx";
import { EndOfInstructions } from "./stages/EndOfInstructions.jsx";
import { TimeoutStage } from "./stages/TimeoutStage.jsx";

export function Stage() {
  const player = usePlayer();
  const players = usePlayers();
  const stage = useStage();

  if (player.stage.get("submit")) {
    if (players.length === 1) {
      return <Loading />;
    }

    switch (stage.get("name")) {
      case "shared_stage":
        return <SharedStage />;
      case "shared_stage_2":
        return <SharedStage_2 />;
      case "shared_stage_3":
        return <SharedStage_3 />;
      case "instructions":
        if (player.stage.get("textStage") == 10) {
          break;
        } else {
          return <Instructions />;
        }
        
    }

    return (
      <div className="text-center text-gray-400 pointer-events-none">
        Please wait for other player(s).
      </div>
    );
  }

  switch (stage.get("name")) {
    case "shared_stage":
      return <SharedStage />;
    case "shared_stage_2":
        return <SharedStage_2 />;
    case "shared_stage_3":
        return <SharedStage_3 />;
    case "instructions":
      return <Instructions />;
    case "end_of_tests":
      return <EndOfTests />;
    case "end_of_instructions":
      return <EndOfInstructions />;
    case "timeout":
      return <TimeoutStage />;
    default:
      return <div>Unknown task</div>;
  }
}
