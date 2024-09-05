import { Chat, useGame, useStage, useRound } from "@empirica/core/player/classic/react";

import React from "react";
import { Profile } from "./Profile";
import { Stage } from "./Stage";

export function Game() {
  const game = useGame();
  const { playerCount } = game.get("treatment");
  const round = useRound();

  return (
    <div className="h-full w-full flex" style={{margin: 0, height: "100%", overflow: "hidden"}}>
      <div className="h-full w-full flex flex-col">
        <Profile />
        <div className="h-full flex items-center justify-center">
          <Stage />
        </div>
      </div>
    </div>
  );
}
