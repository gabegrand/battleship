import React from "react";
import { usePlayer, usePlayers, useRound } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";
import BoardComponent from "../components/BoardComponent.jsx";
import { SpotterBoardComponent } from "../components/SpotterBoardComponent.jsx";

export function InstructionsCaptain() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center", gap:"10px"}}>
      <p style={{fontSize:"18px"}}>In the next screen, you will play as the captain.</p>
      <p style={{fontSize:"18px"}}>The captain can fire at the board, but sees most of it as hidden <b style={{color:"#DfdfDf", textShadow: "-1px 0 black, 0 1px black, 1px 0 black, 0 -1px black"}}>gray</b> tiles.</p>
      <p style={{fontSize:"18px"}}>Here is an example board:</p>

      <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
      <BoardComponent 
      init_tiles={player.stage.get("occTiles")}
      ships={player.stage.get("ships")}
      />
      </div>
      
      <p style={{fontSize:"18px"}}>The captain's role is to fire at the ships (the colored tiles), and ask the spotter questions to gain information to do that.</p>
      <p style={{fontSize:"18px"}}>Click "Continue", and in the next screen, please ask the spotter <b>Is the red ship horizontal?</b> to continue.</p>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => player.stage.set("introStage", 2)}>
        Continue
      </Button>
      </div>
    </div>
  );
}